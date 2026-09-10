"""Shared infrastructure for the 40-zone transient temperature reconstruction project."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml

SENSOR_COUNT = 12
RACK_COUNT = 70
CRAC_COUNT = 46
CONTROL_DIM = RACK_COUNT + 2 * CRAC_COUNT
SAMPLE_RE = re.compile(r"sample_(\d{6})\.npz$")
STRATA = ("zone", "direction", "position_label", "region_size", "variant", "transition_type")
PATH_KEYS = {
    "raw_dir", "samples_dir", "metadata_dir", "parameters_csv", "mesh_npz",
    "zone_cells_npz", "zone_config_json", "zone_summary_json", "sensor_csv",
    "equipment_csv", "zone_graph_csv",
}


def _resolve_path(value: Any, base: Path) -> Any:
    if value in (None, ""):
        return value
    p = Path(str(value)).expanduser()
    return str(p if p.is_absolute() else (base / p).resolve())


def load_config(path: str | Path, smoke: bool = False) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    project = cfg.setdefault("project", {})
    root = Path(_resolve_path(project.get("root_dir", ".."), config_path.parent))
    project["root_dir"] = str(root)
    project["output_dir"] = _resolve_path(project.get("output_dir", "artifacts"), root)
    for key in PATH_KEYS:
        if key in cfg.setdefault("data", {}):
            cfg["data"][key] = _resolve_path(cfg["data"][key], root)
    cfg["_config_path"] = str(config_path)
    cfg["_smoke"] = bool(smoke)
    if smoke:
        s = cfg["smoke"]
        cfg["train"]["epochs"] = int(s["epochs"])
        cfg["train"]["autoencoder_steps_per_epoch"] = int(s["autoencoder_steps_per_epoch"])
        cfg["train"]["validation_examples"] = int(s["validation_examples"])
        cfg["enkf"]["ensemble_size"] = int(s["ensemble_size"])
        cfg["diffusion"]["train_steps"] = int(s["diffusion_train_steps"])
        cfg["diffusion"]["sample_steps"] = int(s["diffusion_sample_steps"])
        cfg["diffusion"]["ensemble_members"] = int(s["diffusion_members"])
        cfg["autoencoder"]["encoder_cell_chunk"] = int(s["cell_chunk"])
        cfg["autoencoder"]["decoder_cell_chunk"] = int(s["cell_chunk"])
        cfg["reconstruct"]["decode_cell_chunk"] = int(s["cell_chunk"])
    return cfg


def output_path(cfg: Mapping[str, Any], *parts: str, mkdir: bool = False) -> Path:
    p = Path(cfg["project"]["output_dir"]).joinpath(*parts)
    if mkdir:
        p.mkdir(parents=True, exist_ok=True)
    return p


def require_paths(cfg: Mapping[str, Any], keys: Sequence[str]) -> None:
    missing = [f"{k}={cfg['data'].get(k)!r}" for k in keys if not cfg["data"].get(k) or not Path(cfg["data"][k]).exists()]
    if missing:
        raise FileNotFoundError("Required paths missing: " + ", ".join(missing))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_csv(path: str | Path) -> List[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def seed_everything(seed: int, rank: int = 0) -> None:
    value = int(seed) + 100003 * int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    distributed: bool = False

    @property
    def main(self) -> bool:
        return self.rank == 0


def init_runtime(device: str = "auto") -> Runtime:
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested = str(device).lower()
    if distributed:
        dev = torch.device("cpu" if requested == "cpu" or not torch.cuda.is_available() else f"cuda:{local_rank}")
    elif requested == "auto":
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    if distributed:
        if dev.type == "cuda":
            torch.cuda.set_device(dev)
        dist.init_process_group(backend="nccl" if dev.type == "cuda" else "gloo")
        return Runtime(dev, dist.get_rank(), dist.get_world_size(), local_rank, True)
    return Runtime(dev)


def barrier(rt: Runtime) -> None:
    if rt.distributed and dist.is_initialized():
        dist.barrier()


def finish_runtime(rt: Runtime) -> None:
    if rt.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def maybe_ddp(model: nn.Module, rt: Runtime) -> nn.Module:
    model = model.to(rt.device)
    if not rt.distributed:
        return model
    kwargs = {"device_ids": [rt.local_rank], "output_device": rt.local_rank} if rt.device.type == "cuda" else {}
    return nn.parallel.DistributedDataParallel(model, **kwargs)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def make_scaler(device: torch.device, enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=enabled and device.type == "cuda")


def distributed_mean(total: float, count: int, rt: Runtime) -> float:
    x = torch.tensor([float(total), float(count)], dtype=torch.float64, device=rt.device)
    if rt.distributed:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return float(x[0].item() / max(x[1].item(), 1.0))


def broadcast_bool(value: bool, rt: Runtime) -> bool:
    x = torch.tensor([int(value)], dtype=torch.int32, device=rt.device)
    if rt.distributed:
        dist.broadcast(x, src=0)
    return bool(x.item())


def epoch_batches(items: Sequence[Any], batch_size: int, seed: int, rt: Runtime) -> Iterator[List[Any]]:
    if not items:
        return
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(items), dtype=np.int64)
    rng.shuffle(order)
    global_batch = max(1, int(batch_size)) * rt.world_size
    padded = int(math.ceil(len(order) / global_batch) * global_batch)
    if padded > len(order):
        order = np.concatenate([order, np.resize(order, padded - len(order))])
    order = order.reshape(-1, rt.world_size, int(batch_size))
    for block in order:
        yield [items[int(i)] for i in block[rt.rank]]


def independent_rows(rows: Sequence[Any], rt: Runtime) -> List[Any]:
    return list(rows)[rt.rank::rt.world_size]


def iter_chunks(length: int, chunk: int) -> Iterator[slice]:
    for start in range(0, int(length), int(chunk)):
        yield slice(start, min(int(length), start + int(chunk)))


def geometry_hash(cfg: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for key in ("mesh_npz", "zone_cells_npz", "zone_config_json", "zone_graph_csv"):
        value = cfg["data"].get(key)
        if value and Path(value).exists():
            p = Path(value)
            st = p.stat()
            h.update(str(p.resolve()).encode())
            h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def save_checkpoint(path: Path, model: nn.Module, optimizer=None, scaler=None, epoch: int = 0,
                    best: float = math.inf, cfg: Optional[Mapping[str, Any]] = None,
                    extra: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch), "best": float(best),
        "geometry_hash": geometry_hash(cfg) if cfg is not None else None,
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, scaler=None,
                    cfg: Optional[Mapping[str, Any]] = None,
                    map_location: str | torch.device = "cpu") -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {path}")
    state = torch.load(path, map_location=map_location, weights_only=False)
    if cfg is not None and state.get("geometry_hash") not in (None, geometry_hash(cfg)):
        raise RuntimeError(f"Checkpoint geometry hash mismatch: {path}")
    unwrap(model).load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return state


def npz_headers(path: str | Path) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
    result = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with zf.open(info) as f:
                version = np.lib.format.read_magic(f)
                reader = np.lib.format.read_array_header_1_0 if version[0] == 1 else np.lib.format.read_array_header_2_0
                shape, _, dtype = reader(f)
            result[info.filename[:-4]] = (tuple(shape), np.dtype(dtype))
    return result


def extract_npz_array(path: Path, key: str, dst: Path,
                      selected_columns: Optional[np.ndarray] = None,
                      output_dtype: Optional[np.dtype] = None) -> Path:
    member = key + ".npy"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        if member not in zf.namelist():
            raise KeyError(f"{path} lacks {key!r}")
        with zf.open(member) as src:
            version = np.lib.format.read_magic(src)
            reader = np.lib.format.read_array_header_1_0 if version[0] == 1 else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(src)
            dtype = np.dtype(dtype)
            if fortran or len(shape) != 2 or dtype.hasobject:
                raise ValueError(f"{path}:{key} must be numeric C-order 2-D, got {shape}/{dtype}")
            cols = None if selected_columns is None else np.asarray(selected_columns, dtype=np.int64)
            dst_dtype = np.dtype(output_dtype) if output_dtype is not None else dtype
            out_shape = shape if cols is None else (shape[0], len(cols))
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            out = np.lib.format.open_memmap(tmp, mode="w+", dtype=dst_dtype, shape=out_shape)
            row_bytes = shape[1] * dtype.itemsize
            for row in range(shape[0]):
                raw = src.read(row_bytes)
                if len(raw) != row_bytes:
                    raise EOFError(f"Truncated payload in {path}:{key}, frame {row}")
                values = np.frombuffer(raw, dtype=dtype, count=shape[1])
                out[row] = (values if cols is None else values[cols]).astype(dst_dtype, copy=False)
            out.flush(); del out
            os.replace(tmp, dst)
    return dst


def _first_key(mapping: Mapping[str, Any], candidates: Sequence[str]) -> Optional[str]:
    return next((k for k in candidates if k in mapping), None)


def hexa_volumes(points: np.ndarray, cells: np.ndarray, chunk: int = 100000) -> np.ndarray:
    faces = np.array([[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]], dtype=np.int64)
    out = np.empty(len(cells), dtype=np.float64)
    for sl in iter_chunks(len(cells), chunk):
        xyz = points[cells[sl]].astype(np.float64)
        center = xyz.mean(axis=1, keepdims=True)
        xyz -= center
        vol = np.zeros(len(xyz), dtype=np.float64)
        for q in faces:
            a,b,c,d = (xyz[:, q[j]] for j in range(4))
            vol += np.einsum("ij,ij->i", a, np.cross(b,c))/6.0
            vol += np.einsum("ij,ij->i", a, np.cross(c,d))/6.0
        out[sl] = np.abs(vol)
    if not np.isfinite(out).all() or np.any(out <= 0):
        raise ValueError("Non-positive/non-finite cell volumes")
    return out


def _load_zone_ids(cfg: Mapping[str, Any], mesh: Mapping[str, np.ndarray], n: int) -> Tuple[np.ndarray, int]:
    # Prefer the explicitly supplied connected-zone file over any stale zone IDs in mesh.npz.
    zone_path = cfg["data"].get("zone_cells_npz")
    if zone_path and Path(zone_path).exists():
        with np.load(zone_path, allow_pickle=False) as z:
            key = _first_key(z, ("zone_id", "cell_zone_ids", "cell_zone_id"))
            if key is None:
                raise KeyError(f"No zone_id array in {zone_path}; found {z.files}")
            zone_id = np.asarray(z[key])
    else:
        key = _first_key(mesh, ("cell_zone_ids", "cell_zone_id", "zone_id"))
        if key is None:
            raise FileNotFoundError("No zone IDs in connected-zone file or mesh.npz")
        zone_id = np.asarray(mesh[key])
    if zone_id.shape != (n,):
        raise ValueError(f"zone_id shape must be ({n},), got {zone_id.shape}")
    active = np.unique(zone_id[zone_id > 0]).astype(int)
    if len(active) == 0 or not np.array_equal(active, np.arange(1, int(active.max()) + 1)):
        raise ValueError(f"Active zone IDs must be contiguous 1..K, found {active.tolist()}")
    num_zones = int(active.max())
    expected = int(cfg["data"].get("expected_fluid_zones", num_zones))
    if num_zones != expected:
        raise ValueError(f"Connected zone file has K={num_zones}, config expects {expected}")
    if np.count_nonzero(zone_id == 0):
        raise ValueError("zone_id contains outside/unassigned fluid cells (zone 0)")
    return zone_id.astype(np.int16), num_zones


def select_smoke_cells(zone_id: np.ndarray, limit: int, seed: int, num_zones: int) -> np.ndarray:
    if limit <= 0 or limit >= len(zone_id):
        return np.arange(len(zone_id), dtype=np.int64)
    rng = np.random.default_rng(seed)
    per = max(1, limit // num_zones)
    parts = []
    for zid in range(1, num_zones + 1):
        idx = np.flatnonzero(zone_id == zid)
        parts.append(rng.choice(idx, min(per, len(idx)), replace=False))
    out = np.unique(np.concatenate(parts))
    if len(out) < limit:
        rest = np.setdiff1d(np.arange(len(zone_id)), out, assume_unique=True)
        out = np.concatenate([out, rng.choice(rest, min(limit-len(out), len(rest)), replace=False)])
    return np.sort(out[:limit]).astype(np.int64)


def normalize_rows(a: np.ndarray) -> np.ndarray:
    return (a / np.maximum(a.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _quad_area(xyz: np.ndarray) -> float:
    a,b,c,d = xyz.astype(np.float64)
    return 0.5*np.linalg.norm(np.cross(b-a,c-a)) + 0.5*np.linalg.norm(np.cross(c-a,d-a))


def shared_face_zone_graph(cells: np.ndarray, zone_id: np.ndarray, num_zones: int,
                           points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if cells.ndim != 2 or cells.shape[1] != 8:
        raise ValueError(f"Expected hexahedral cells [N,8], got {cells.shape}")
    face_lut = np.array([[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]], dtype=np.int64)
    faces = np.sort(cells[:, face_lut].reshape(-1,4), axis=1)
    owners = np.repeat(zone_id.astype(np.int32), 6)
    order = np.lexsort((faces[:,3],faces[:,2],faces[:,1],faces[:,0]))
    faces, owners = faces[order], owners[order]
    pair_pos = np.flatnonzero(np.all(faces[1:] == faces[:-1], axis=1))
    w = np.zeros((num_zones, num_zones), dtype=np.float64)
    for pos in pair_pos:
        zi,zj = int(owners[pos]), int(owners[pos+1])
        if zi == zj:
            continue
        area = _quad_area(points[faces[pos]]) if points is not None else 1.0
        w[zi-1,zj-1] += area
        w[zj-1,zi-1] += area
    scale = max(float(w.sum(axis=1).max()), 1e-12)
    exchange = (w / scale).astype(np.float32)
    return normalize_rows(w.astype(np.float32)), exchange


def load_zone_graph_csv(path: str | Path, num_zones: int) -> Tuple[np.ndarray, np.ndarray]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"Empty zone graph CSV: {path}")
    keys = rows[0].keys()
    skey = next((k for k in ("src","source","zone_i","zone_u","from","i") if k in keys), None)
    dkey = next((k for k in ("dst","target","zone_j","zone_v","to","j") if k in keys), None)
    wkey = next((k for k in ("weight","area","shared_area","shared_face_area","w") if k in keys), None)
    if skey is None or dkey is None:
        raise KeyError("Cannot identify graph source/destination columns")
    w = np.zeros((num_zones,num_zones), dtype=np.float64)
    for row in rows:
        i,j = int(row[skey]), int(row[dkey])
        if not (1 <= i <= num_zones and 1 <= j <= num_zones) or i == j:
            continue
        value = float(row[wkey]) if wkey and row.get(wkey) not in (None,"") else 1.0
        w[i-1,j-1] += value; w[j-1,i-1] += value
    scale = max(float(w.sum(axis=1).max()), 1e-12)
    return normalize_rows(w.astype(np.float32)), (w/scale).astype(np.float32)


def local_coordinates(centers: np.ndarray, zone_id: np.ndarray, num_zones: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    out = np.empty_like(centers, dtype=np.float32)
    lo = np.empty((num_zones,3), dtype=np.float32)
    hi = np.empty((num_zones,3), dtype=np.float32)
    for zid in range(1,num_zones+1):
        idx = np.flatnonzero(zone_id == zid)
        lo[zid-1] = centers[idx].min(axis=0)
        hi[zid-1] = centers[idx].max(axis=0)
        out[idx] = 2.0*(centers[idx]-lo[zid-1])/np.maximum(hi[zid-1]-lo[zid-1],1e-6)-1.0
    return out.astype(np.float32),lo,hi


def load_geometry(cfg: Mapping[str, Any], smoke: bool = False) -> Dict[str,np.ndarray]:
    require_paths(cfg, ["mesh_npz", "zone_config_json", "sensor_csv", "equipment_csv"])
    with np.load(cfg["data"]["mesh_npz"], allow_pickle=False) as m:
        mesh = {k:m[k] for k in m.files}
    ckey = _first_key(mesh, ("cell_centers_m","cell_centers","centers"))
    if ckey is None:
        raise KeyError(f"mesh.npz lacks cell centers; found {sorted(mesh)}")
    centers_all = np.asarray(mesh[ckey], dtype=np.float32)
    n = len(centers_all)
    expected = int(cfg["data"].get("expected_cells", n))
    if n != expected:
        raise ValueError(f"mesh has {n} cells; config expects {expected}")
    zone_all,num_zones = _load_zone_ids(cfg, mesh, n)
    vkey = _first_key(mesh, ("cell_volumes_m3","cell_volumes","volumes"))
    points = np.asarray(mesh["points"]) if "points" in mesh else None
    cells_all = np.asarray(mesh["cells"]) if "cells" in mesh else None
    if vkey is not None:
        volumes_all = np.asarray(mesh[vkey], dtype=np.float64)
    elif points is not None and cells_all is not None:
        volumes_all = hexa_volumes(points,cells_all,int(cfg["data"].get("cell_chunk",100000)))
    else:
        raise KeyError("mesh.npz must contain volumes or points+cells")
    graph_csv = cfg["data"].get("zone_graph_csv")
    if graph_csv and Path(graph_csv).exists():
        adjacency,exchange = load_zone_graph_csv(graph_csv,num_zones)
        graph_source = "zone_graph_csv"
    else:
        if cells_all is None:
            raise RuntimeError("No zone_graph_csv and mesh has no cell connectivity")
        adjacency,exchange = shared_face_zone_graph(cells_all,zone_all,num_zones,points)
        graph_source = "mesh_shared_faces"
    limit = int(cfg["smoke"]["max_cells"]) if smoke else 0
    idx = select_smoke_cells(zone_all,limit,int(cfg["project"]["seed"]),num_zones)
    centers,zone_id,volumes = centers_all[idx],zone_all[idx],volumes_all[idx]
    local,zone_lo,zone_hi = local_coordinates(centers,zone_id,num_zones)
    order = np.argsort(zone_id, kind="stable")
    counts = np.bincount(zone_id,minlength=num_zones+1)[1:]
    offsets = np.concatenate([[0],np.cumsum(counts)]).astype(np.int64)
    zone_volumes = np.bincount(zone_id,weights=volumes,minlength=num_zones+1)[1:].astype(np.float64)
    zone_centers = np.empty((num_zones,3),dtype=np.float64)
    for zid in range(1,num_zones+1):
        zi = np.flatnonzero(zone_id==zid)
        zone_centers[zid-1] = np.average(centers[zi],axis=0,weights=volumes[zi])
    return {
        "centers": centers.astype(np.float32), "local_coords": local,
        "zone_id": zone_id.astype(np.int16), "volumes": volumes.astype(np.float64),
        "cell_indices": idx.astype(np.int64), "num_zones": np.int32(num_zones),
        "zone_order": order.astype(np.int64), "zone_offsets": offsets,
        "zone_volumes": zone_volumes, "zone_centers": zone_centers.astype(np.float32),
        "zone_lo": zone_lo, "zone_hi": zone_hi,
        "adjacency": adjacency, "exchange_weights": exchange,
        "graph_source": np.asarray(graph_source),
    }


def save_geometry(path: Path, geo: Mapping[str,Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k:v for k,v in geo.items() if isinstance(v,np.ndarray) or np.isscalar(v)})


def num_zones_from_geo(geo: Mapping[str,np.ndarray]) -> int:
    return int(np.asarray(geo["num_zones"]).item())


def zone_reduce(values: np.ndarray, zone_id: np.ndarray, volumes: np.ndarray,
                num_zones: Optional[int] = None) -> np.ndarray:
    x = np.asarray(values)
    k = int(num_zones or np.max(zone_id))
    flat = x.reshape(-1,x.shape[-1]).astype(np.float64,copy=False)
    denom = np.bincount(zone_id,weights=volumes,minlength=k+1)[1:]
    out = np.empty((len(flat),k),dtype=np.float64)
    for i,row in enumerate(flat):
        out[i] = np.bincount(zone_id,weights=volumes*row,minlength=k+1)[1:]/denom
    return out.reshape(x.shape[:-1]+(k,)).astype(np.float32)


def zone_lift(z: np.ndarray, zone_id: np.ndarray) -> np.ndarray:
    return np.asarray(z)[...,zone_id.astype(np.int64)-1]


def project_zero_zone_mean(residual: np.ndarray, zone_id: np.ndarray,
                           volumes: np.ndarray, num_zones: Optional[int]=None) -> np.ndarray:
    return residual-zone_lift(zone_reduce(residual,zone_id,volumes,num_zones),zone_id)


def read_zone_labels(cfg: Mapping[str,Any]) -> Dict[int,str]:
    with open(cfg["data"]["zone_config_json"],"r",encoding="utf-8-sig") as f:
        raw=json.load(f)["labels"]
    return {int(k):str(v) for k,v in raw.items()}


def device_zone_mapping(cfg: Mapping[str,Any], centers: np.ndarray, zone_id: np.ndarray,
                        volumes: np.ndarray, num_zones: int) -> Tuple[np.ndarray,List[dict]]:
    rows=read_csv(cfg["data"]["equipment_csv"])
    racks=sorted(int(r["number"]) for r in rows if r["device_type"]=="Rack")
    cracs=sorted(int(r["number"]) for r in rows if r["device_type"]=="CRAC")
    if racks != list(range(1,RACK_COUNT+1)) or cracs != list(range(1,CRAC_COUNT+1)):
        raise ValueError("Equipment CSV must contain Rack 1..70 and CRAC 1..46")
    ordered=sorted(rows,key=lambda r:(0 if r["device_type"]=="Rack" else 1,int(r["number"])))
    shell=float(cfg["geometry"]["device_shell_m"]); minimum=int(cfg["geometry"]["device_mapping_min_cells"])
    mapping=np.zeros((len(ordered),num_zones),dtype=np.float32)
    for j,row in enumerate(ordered):
        lo=np.array([float(row["x_min_m"]),float(row["y_min_m"]),float(row["z_min_m"])])
        hi=np.array([float(row["x_max_m"]),float(row["y_max_m"]),float(row["z_max_m"])])
        outer=np.all((centers>=lo-shell)&(centers<=hi+shell),axis=1)
        inner=np.all((centers>=lo)&(centers<=hi),axis=1)
        ids=np.flatnonzero(outer&~inner)
        if len(ids)<minimum:
            c=np.array([float(row["x_center_m"]),float(row["y_center_m"]),float(row["z_center_m"])])
            take=min(minimum,len(centers))
            ids=np.argpartition(np.sum((centers-c)**2,axis=1),take-1)[:take]
        w=np.bincount(zone_id[ids],weights=volumes[ids],minlength=num_zones+1)[1:]
        mapping[j]=w/max(float(w.sum()),1e-12)
    return mapping,ordered


def sensor_mapping(cfg: Mapping[str,Any], centers: np.ndarray, zone_id: np.ndarray,
                   num_zones: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray,List[dict]]:
    rows=read_csv(cfg["data"]["sensor_csv"])
    if len(rows)!=SENSOR_COUNT:
        raise ValueError(f"Sensor CSV must contain {SENSOR_COUNT} rows, found {len(rows)}")
    label_to_id={name:zid for zid,name in read_zone_labels(cfg).items()}
    local=[]; sensor_zone=[]; h=np.zeros((SENSOR_COUNT,num_zones),dtype=np.float32)
    for j,row in enumerate(rows):
        state=row["zone_state"]
        if state not in label_to_id:
            raise ValueError(f"Sensor {row['sensor_id']} references unknown zone_state={state!r}")
        zid=label_to_id[state]
        if not 1<=zid<=num_zones:
            raise ValueError(f"Sensor maps to non-fluid zone {zid}")
        candidates=np.flatnonzero(zone_id==zid)
        xyz=np.array([float(row["x_m"]),float(row["y_m"]),float(row["z_m"])])
        chosen=candidates[np.argmin(np.sum((centers[candidates]-xyz)**2,axis=1))]
        local.append(int(chosen)); sensor_zone.append(zid); h[j,zid-1]=1.0
    return np.asarray(local),np.asarray(sensor_zone),h,rows


def control_sequences(row: Mapping[str,str], frames: int) -> Tuple[np.ndarray,np.ndarray]:
    p0=np.array([float(row[f"P0_Rack{i}_kW"]) for i in range(1,RACK_COUNT+1)],dtype=np.float32)
    p1=np.array([float(row[f"P1_Rack{i}_kW"]) for i in range(1,RACK_COUNT+1)],dtype=np.float32)
    t0=np.array([float(row[f"AC0_CRAC{i}_Temp_C"]) for i in range(1,CRAC_COUNT+1)],dtype=np.float32)
    t1=np.array([float(row[f"AC1_CRAC{i}_Temp_C"]) for i in range(1,CRAC_COUNT+1)],dtype=np.float32)
    f0=np.array([float(row[f"AC0_CRAC{i}_Fan_pct"]) for i in range(1,CRAC_COUNT+1)],dtype=np.float32)
    f1=np.array([float(row[f"AC1_CRAC{i}_Fan_pct"]) for i in range(1,CRAC_COUNT+1)],dtype=np.float32)
    u0=np.concatenate([p0,t0,f0]); u1=np.concatenate([p1,t1,f1])
    frame=np.repeat(u1[None],frames,axis=0); frame[0]=u0
    transition=np.repeat(u1[None],max(frames-1,0),axis=0)
    return frame.astype(np.float32),transition.astype(np.float32)


def greedy_stratified_split(rows: List[dict], fractions: Mapping[str,float], seed: int) -> Dict[str,List[int]]:
    names=["train","validation","test"]; n=len(rows)
    target={k:int(round(float(fractions[k])*n)) for k in names}; target["train"]+=n-sum(target.values())
    labels=[[f"{key}={row.get(key,'')}" for key in STRATA] for row in rows]
    totals={}
    for item in labels:
        for label in item: totals[label]=totals.get(label,0)+1
    rng=random.Random(int(seed)); order=list(range(n)); rng.shuffle(order)
    order.sort(key=lambda i:sum(1.0/totals[x] for x in labels[i]),reverse=True)
    assigned={k:[] for k in names}; counts={k:{} for k in names}
    for i in order:
        choices=[k for k in names if len(assigned[k])<target[k]]
        def cost(name):
            lc=sum((counts[name].get(label,0)+1)/max(totals[label]*float(fractions[name]),1.0) for label in labels[i])
            return lc+0.1*(len(assigned[name])+1)/max(target[name],1)
        chosen=min(choices,key=cost); sid=int(rows[i]["sample_id"]); assigned[chosen].append(sid)
        for label in labels[i]: counts[chosen][label]=counts[chosen].get(label,0)+1
    for name in names: assigned[name].sort()
    return assigned


def scan_manifest(cfg: Mapping[str,Any]) -> Tuple[List[dict],Dict[str,List[int]]]:
    require_paths(cfg,["samples_dir","parameters_csv"])
    params={int(r["sample_id"]):r for r in read_csv(cfg["data"]["parameters_csv"])}
    files=[]; ef=int(cfg["data"]["expected_frames"]); ec=int(cfg["data"]["expected_cells"])
    time_key=cfg["data"]["time_key"]; temp_key=cfg["data"]["temperature_key"]
    metadata_dir=cfg["data"].get("metadata_dir")
    for path in sorted(Path(cfg["data"]["samples_dir"]).glob("sample_*.npz")):
        m=SAMPLE_RE.match(path.name)
        if not m: continue
        sid=int(m.group(1))
        if sid not in params: raise FileNotFoundError(f"Sample {sid} lacks parameters row")
        headers=npz_headers(path)
        if headers.get(time_key,(None,None))[0]!=(ef,): raise ValueError(f"{path}:{time_key} wrong shape")
        if headers.get(temp_key,(None,None))[0]!=(ef,ec): raise ValueError(f"{path}:{temp_key} wrong shape")
        row=dict(params[sid]); row.update({"sample_id":sid,"sample_path":str(path.resolve())})
        if metadata_dir:
            meta=Path(metadata_dir)/f"sample_{sid:06d}_metadata.json"
            row["metadata_path"]=str(meta.resolve()) if meta.exists() else ""
        files.append(row)
    if not files: raise FileNotFoundError(f"No sample_*.npz in {cfg['data']['samples_dir']}")
    split=greedy_stratified_split(files,cfg["split"],int(cfg["project"]["seed"]))
    membership={sid:name for name,ids in split.items() for sid in ids}
    for row in files: row["split"]=membership[int(row["sample_id"])]
    return files,split


def write_manifest(path: Path, rows: List[dict]) -> None:
    fields=["sample_id","split","sample_path","metadata_path"]+[k for k in ("case_id","zone","direction","position_label","region_size","variant","transition_type") if k in rows[0]]
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def read_manifest(cfg: Mapping[str,Any]) -> List[dict]:
    path=output_path(cfg,"prepared","manifest.csv")
    if not path.exists(): raise FileNotFoundError(f"Manifest missing: {path}")
    rows=read_csv(path)
    for row in rows: row["sample_id"]=int(row["sample_id"])
    return rows


def prepared_sample_path(cfg: Mapping[str,Any], sample_id: int) -> Path:
    return output_path(cfg,"prepared","trajectories",f"sample_{sample_id:06d}.npz")


def temperature_cache_path(cfg: Mapping[str,Any], sample_id: int) -> Path:
    return output_path(cfg,"prepared","temperature",f"sample_{sample_id:06d}.npy")


def load_prepared(cfg: Mapping[str,Any], sample_id: int) -> Dict[str,np.ndarray]:
    with np.load(prepared_sample_path(cfg,sample_id),allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def load_prepared_geometry(cfg: Mapping[str,Any]) -> Dict[str,np.ndarray]:
    path=output_path(cfg,"prepared","geometry.npz")
    if not path.exists(): raise FileNotFoundError(f"Prepared geometry missing: {path}")
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


@dataclass
class LowDimNorm:
    z_mean: np.ndarray; z_std: np.ndarray
    u_frame_mean: np.ndarray; u_frame_std: np.ndarray
    u_transition_mean: np.ndarray; u_transition_std: np.ndarray
    y_mean: np.ndarray; y_std: np.ndarray
    physics_scales: np.ndarray

    @classmethod
    def load(cls,cfg: Mapping[str,Any]) -> "LowDimNorm":
        with np.load(output_path(cfg,"prepared","normalization_train_only.npz")) as z:
            return cls(**{name:z[name] for name in cls.__annotations__})
    def norm_z(self,x): return ((x-self.z_mean)/self.z_std).astype(np.float32)
    def denorm_z(self,x): return (x*self.z_std+self.z_mean).astype(np.float32)
    def norm_frame_u(self,x): return ((x-self.u_frame_mean)/self.u_frame_std).astype(np.float32)
    def norm_transition_u(self,x): return ((x-self.u_transition_mean)/self.u_transition_std).astype(np.float32)
    def norm_innovation(self,x): return (x/self.y_std).astype(np.float32)
    def norm_pdiag(self,x): return np.log1p(np.maximum(x,0.0)/(self.z_std**2)).astype(np.float32)


def history_window(a: np.ndarray, end: int, history: int) -> np.ndarray:
    if len(a)==0: raise ValueError("Empty history sequence")
    end=min(max(int(end),0),len(a)-1); start=max(0,end-int(history)+1); x=a[start:end+1]
    if len(x)<history: x=np.concatenate([np.repeat(x[:1],history-len(x),axis=0),x],axis=0)
    return x


def physics_features_np(z: np.ndarray, u: np.ndarray, exchange: np.ndarray,
                        device_zone: np.ndarray, zone_volumes: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    z=np.asarray(z,dtype=np.float64); u=np.asarray(u,dtype=np.float64)
    vr=np.asarray(zone_volumes,dtype=np.float64)/max(float(np.mean(zone_volumes)),1e-12)
    row=exchange.sum(axis=1)
    transport=(np.einsum("...j,ij->...i",z,exchange)-z*row)/vr
    rack=np.einsum("...r,rk->...k",u[...,:RACK_COUNT],device_zone[:RACK_COUNT])/vr
    temp=u[...,RACK_COUNT:RACK_COUNT+CRAC_COUNT]
    fan=np.clip(u[...,RACK_COUNT+CRAC_COUNT:]/100.0,0.0,None)
    cmap=device_zone[RACK_COUNT:]
    fan_zone=np.einsum("...c,ck->...k",fan,cmap)
    supply_num=np.einsum("...c,ck->...k",fan*temp,cmap)
    supply=supply_num/np.maximum(fan_zone,1e-6)
    cooling=fan_zone*np.maximum(z-supply,0.0)/vr
    return transport.astype(np.float32),rack.astype(np.float32),cooling.astype(np.float32)


class GraphBlock(nn.Module):
    def __init__(self,dim:int):
        super().__init__(); self.self_fc=nn.Linear(dim,dim); self.neighbor_fc=nn.Linear(dim,dim,bias=False); self.norm=nn.LayerNorm(dim)
    def forward(self,x,adjacency):
        neighbor=torch.einsum("ij,bjd->bid",adjacency,x)
        return self.norm(x+F.silu(self.self_fc(x)+self.neighbor_fc(neighbor)))


class CausalTemporalBlock(nn.Module):
    def __init__(self,dim:int,dilation:int):
        super().__init__(); self.dilation=int(dilation); self.conv1=nn.Conv1d(dim,dim,3,dilation=dilation); self.conv2=nn.Conv1d(dim,dim,3,dilation=dilation); self.norm=nn.GroupNorm(1,dim)
    def forward(self,x):
        pad=2*self.dilation
        y=F.silu(self.conv1(F.pad(x,(pad,0))))
        y=self.conv2(F.pad(y,(pad,0)))
        return self.norm(x+y)


class LowParameterZoneDynamics(nn.Module):
    """Three shared positive gains plus a bounded shared TCN-graph residual."""
    def __init__(self, geo: Mapping[str,np.ndarray], norm: LowDimNorm,
                 hidden: int, tcn_blocks: int, graph_blocks: int,
                 residual_max_C: float, initial_gains_C: Sequence[float]):
        super().__init__()
        k=num_zones_from_geo(geo); self.num_zones=k; self.residual_max_C=float(residual_max_C)
        self.register_buffer("device_zone",torch.as_tensor(geo["device_zone"],dtype=torch.float32))
        self.register_buffer("exchange",torch.as_tensor(geo["exchange_weights"],dtype=torch.float32))
        self.register_buffer("adjacency",torch.as_tensor(geo["adjacency"],dtype=torch.float32))
        self.register_buffer("zone_volume_ratio",torch.as_tensor(geo["zone_volumes"]/np.mean(geo["zone_volumes"]),dtype=torch.float32))
        self.register_buffer("z_mean",torch.as_tensor(norm.z_mean,dtype=torch.float32))
        self.register_buffer("z_std",torch.as_tensor(norm.z_std,dtype=torch.float32))
        self.register_buffer("physics_scales",torch.as_tensor(norm.physics_scales,dtype=torch.float32))
        init=torch.as_tensor(initial_gains_C,dtype=torch.float32).clamp_min(1e-5)
        self.raw_beta=nn.Parameter(torch.log(torch.expm1(init)))
        self.input=nn.Linear(4,hidden)
        self.temporal=nn.ModuleList([CausalTemporalBlock(hidden,2**i) for i in range(tcn_blocks)])
        self.graph=nn.ModuleList([GraphBlock(hidden) for _ in range(graph_blocks)])
        self.residual_head=nn.Linear(hidden,1)

    def features(self,z,u):
        vr=self.zone_volume_ratio
        row=self.exchange.sum(dim=1)
        transport=(torch.einsum("...j,ij->...i",z,self.exchange)-z*row)/vr
        rack=torch.einsum("...r,rk->...k",u[...,:RACK_COUNT],self.device_zone[:RACK_COUNT])/vr
        temp=u[...,RACK_COUNT:RACK_COUNT+CRAC_COUNT]
        fan=torch.clamp(u[...,RACK_COUNT+CRAC_COUNT:]/100.0,min=0.0)
        cmap=self.device_zone[RACK_COUNT:]
        fan_zone=torch.einsum("...c,ck->...k",fan,cmap)
        supply=torch.einsum("...c,ck->...k",fan*temp,cmap)/fan_zone.clamp_min(1e-6)
        cooling=fan_zone*F.relu(z-supply)/vr
        return transport,rack,cooling

    def forward(self,z_history,controls,return_terms:bool=False):
        # Physical units: z in degC, controls in kW/degC/%.
        trans,rack,cool=self.features(z_history,controls)
        scales=self.physics_scales.clamp_min(1e-6)
        z_norm=(z_history-self.z_mean)/self.z_std
        x=torch.stack([z_norm,trans/scales[0],rack/scales[1],cool/scales[2]],dim=-1)
        b,h,k,_=x.shape
        x=self.input(x).permute(0,2,3,1).reshape(b*k,-1,h)
        for block in self.temporal: x=block(x)
        x=x[...,-1].reshape(b,k,-1)
        for block in self.graph: x=block(x,self.adjacency)
        residual=self.residual_max_C*torch.tanh(self.residual_head(x).squeeze(-1))
        beta=F.softplus(self.raw_beta)
        core=beta[0]*(trans[:,-1]/scales[0])+beta[1]*(rack[:,-1]/scales[1])-beta[2]*(cool[:,-1]/scales[2])
        pred=z_history[:,-1]+core+residual
        if return_terms:
            return pred,{"transport":beta[0]*(trans[:,-1]/scales[0]),"rack":beta[1]*(rack[:,-1]/scales[1]),"crac":-beta[2]*(cool[:,-1]/scales[2]),"residual":residual,"beta":beta}
        return pred


def coordinate_features(coords: torch.Tensor, frequencies: int) -> torch.Tensor:
    features=[coords]
    for k in range(int(frequencies)):
        omega=(2.0**k)*math.pi; features.extend([torch.sin(coords*omega),torch.cos(coords*omega)])
    return torch.cat(features,dim=-1)


class ZonedFieldAutoencoder(nn.Module):
    """Shared full-zone PointNet/DeepSets encoder and coordinate implicit decoder."""
    def __init__(self,num_zones:int,latent_dim:int,point_hidden:int,decoder_hidden:int,
                 frequencies:int,zone_embedding_dim:int):
        super().__init__(); self.num_zones=int(num_zones); self.latent_dim=int(latent_dim); self.frequencies=int(frequencies)
        coord_dim=3*(1+2*self.frequencies)
        self.zone_embed=nn.Embedding(self.num_zones,int(zone_embedding_dim))
        self.point_encoder=nn.Sequential(
            nn.Linear(coord_dim+2,point_hidden),nn.SiLU(),
            nn.Linear(point_hidden,point_hidden),nn.SiLU(),
            nn.Linear(point_hidden,point_hidden),nn.SiLU(),
        )
        self.to_latent=nn.Sequential(
            nn.Linear(2*point_hidden+zone_embedding_dim,point_hidden),nn.SiLU(),
            nn.Linear(point_hidden,self.latent_dim),
        )
        self.decoder=nn.Sequential(
            nn.Linear(coord_dim+self.latent_dim+zone_embedding_dim,decoder_hidden),nn.SiLU(),
            nn.Linear(decoder_hidden,decoder_hidden),nn.SiLU(),
            nn.Linear(decoder_hidden,decoder_hidden),nn.SiLU(),
            nn.Linear(decoder_hidden,1),
        )

    def encode_zone(self,values,coords,volumes,zone_index:int,chunk:int) -> torch.Tensor:
        if values.ndim!=1: raise ValueError("encode_zone expects values [N]")
        total_w=volumes.sum().clamp_min(1e-12); weighted_sum=None; maxima=[]
        mean_v=volumes.mean().clamp_min(1e-12)
        for sl in iter_chunks(len(values),chunk):
            p=coordinate_features(coords[sl],self.frequencies)
            logv=torch.log((volumes[sl]/mean_v).clamp_min(1e-8)).unsqueeze(-1)
            x=self.point_encoder(torch.cat([p,values[sl,None],logv],dim=-1))
            part=(x*volumes[sl,None]).sum(dim=0)
            weighted_sum=part if weighted_sum is None else weighted_sum+part
            maxima.append(x.max(dim=0).values)
        pooled=torch.cat([weighted_sum/total_w,torch.stack(maxima).max(dim=0).values],dim=-1)
        zid=torch.tensor([int(zone_index)],dtype=torch.long,device=values.device)
        return self.to_latent(torch.cat([pooled[None],self.zone_embed(zid)],dim=-1)).squeeze(0)

    def decode_zone(self,latent,coords,zone_index:int) -> torch.Tensor:
        if latent.ndim==1: latent=latent[None]
        b=latent.shape[0]; p=coordinate_features(coords,self.frequencies)
        zid=torch.full((len(coords),),int(zone_index),dtype=torch.long,device=coords.device)
        e=self.zone_embed(zid)
        p=p[None].expand(b,-1,-1); e=e[None].expand(b,-1,-1); h=latent[:,None,:].expand(-1,len(coords),-1)
        return self.decoder(torch.cat([p,h,e],dim=-1)).squeeze(-1)

    def decode(self,latent,coords,zones):
        b=latent.shape[0]; p=coordinate_features(coords,self.frequencies)
        local=latent[:,zones]; e=self.zone_embed(zones)[None].expand(b,-1,-1)
        return self.decoder(torch.cat([p[None].expand(b,-1,-1),local,e],dim=-1)).squeeze(-1)

    def forward(self,values,coords,volumes,zone_index:int,encoder_chunk:int,decoder_chunk:int):
        latent=self.encode_zone(values,coords,volumes,int(zone_index),int(encoder_chunk))
        parts=[self.decode_zone(latent,coords[sl],int(zone_index)) for sl in iter_chunks(len(values),int(decoder_chunk))]
        return torch.cat(parts,dim=1).squeeze(0),latent


class ConditionEncoder(nn.Module):
    def __init__(self,history:int,hidden:int,device_zone:np.ndarray,num_zones:int,graph_blocks:int=1):
        super().__init__(); self.history=int(history); self.num_zones=int(num_zones)
        self.register_buffer("device_zone",torch.as_tensor(device_zone,dtype=torch.float32))
        per_zone=2+3+SENSOR_COUNT
        self.net=nn.Sequential(nn.Linear(self.history*per_zone,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.graph=nn.ModuleList([GraphBlock(hidden) for _ in range(graph_blocks)])
    def forward(self,z,pdiag,u,innovation,h_zone,adjacency):
        rack=u[...,:RACK_COUNT]@self.device_zone[:RACK_COUNT]
        temp=u[...,RACK_COUNT:RACK_COUNT+CRAC_COUNT]@self.device_zone[RACK_COUNT:]
        fan=u[...,RACK_COUNT+CRAC_COUNT:]@self.device_zone[RACK_COUNT:]
        obs=innovation[...,None,:]*h_zone.T[None,None,:,:]
        x=torch.cat([z[...,None],pdiag[...,None],rack[...,None],temp[...,None],fan[...,None],obs],dim=-1)
        x=self.net(x.transpose(1,2).flatten(2))
        for block in self.graph: x=block(x,adjacency)
        return x


class DeterministicSR(nn.Module):
    def __init__(self,condition:ConditionEncoder,latent_dim:int):
        super().__init__(); hidden=condition.net[-1].out_features; self.condition=condition
        self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,latent_dim))
    def forward(self,z,pdiag,u,innovation,h_zone,adjacency):
        return self.head(self.condition(z,pdiag,u,innovation,h_zone,adjacency))


class DiffusionSR(nn.Module):
    def __init__(self,condition:ConditionEncoder,latent_dim:int,hidden:int):
        super().__init__(); self.condition=condition
        condition_hidden=condition.net[-1].out_features
        self.time=nn.Sequential(nn.Linear(1,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.in_mlp=nn.Sequential(nn.Linear(latent_dim+condition_hidden+hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
        self.graph=GraphBlock(hidden); self.out=nn.Linear(hidden,latent_dim)
    def forward(self,noisy,t,z,pdiag,u,innovation,h_zone,adjacency):
        c=self.condition(z,pdiag,u,innovation,h_zone,adjacency)
        te=self.time(t[:,None].float())[:,None].expand(-1,noisy.shape[1],-1)
        x=self.in_mlp(torch.cat([noisy,c,te],dim=-1)); x=self.graph(x,adjacency)
        return self.out(x)


def diffusion_schedule(steps:int,beta_start:float,beta_end:float,device:torch.device):
    beta=torch.linspace(float(beta_start),float(beta_end),int(steps),device=device)
    return beta,torch.cumprod(1.0-beta,dim=0)


@torch.no_grad()
def ddim_sample(model:nn.Module,shape:Tuple[int,int,int],condition:tuple,sample_steps:int,
                train_steps:int,beta_start:float,beta_end:float,device:torch.device,
                generator:Optional[torch.Generator]=None):
    _,alpha_bar=diffusion_schedule(train_steps,beta_start,beta_end,device)
    indices=torch.linspace(train_steps-1,0,sample_steps,device=device).round().long().unique_consecutive()
    x=torch.randn(shape,device=device,generator=generator)
    for j,ti in enumerate(indices):
        t=torch.full((shape[0],),int(ti),device=device,dtype=torch.long)
        eps=model(x,t.float()/max(train_steps-1,1),*condition)
        at=alpha_bar[ti]; x0=(x-torch.sqrt(1-at)*eps)/torch.sqrt(at)
        if j==len(indices)-1: x=x0
        else:
            ap=alpha_bar[indices[j+1]]; x=torch.sqrt(ap)*x0+torch.sqrt(1-ap)*eps
    return x


def cli_common(parser) -> None:
    parser.add_argument("--config",default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--device",default="auto",help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--seed",type=int,default=None)
    parser.add_argument("--num_workers",type=int,default=None)
    parser.add_argument("--smoke",action="store_true")


def configure_args(args) -> Tuple[dict,Runtime]:
    cfg=load_config(args.config,args.smoke)
    if args.seed is not None: cfg["project"]["seed"]=int(args.seed)
    if args.num_workers is not None: cfg["train"]["num_workers"]=int(args.num_workers)
    rt=init_runtime(args.device); seed_everything(int(cfg["project"]["seed"]),rt.rank)
    return cfg,rt
