"""Shared model loading and inference for evaluation, paper plots, and benchmarking."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from common import (
    ConditionEncoder, DeterministicSR, DiffusionSR, LowDimNorm,
    ZonedFieldAutoencoder, ddim_sample, history_window, iter_chunks,
    load_checkpoint, num_zones_from_geo, output_path, read_manifest, zone_lift,
)

METHODS = ("zone_const", "4dsrda", "diffsrda", "proposed")
STOCHASTIC_METHODS = {"diffsrda", "proposed"}
DISPLAY = {
    "zone_const": "Zone-Analysis-Const", "4dsrda": "4D-SRDA",
    "diffsrda": "DiffSRDA", "proposed": "Proposed",
}


@dataclass
class ModelBundle:
    method: str
    ae: torch.nn.Module | None = None
    sr: torch.nn.Module | None = None
    latent_mean: np.ndarray | None = None
    latent_std: np.ndarray | None = None
    field_mean: float | None = None
    field_std: float | None = None


def target_name(method: str) -> str:
    return "residual" if method == "proposed" else "full"


def make_autoencoder(cfg, geo, target: str, device: torch.device):
    a = cfg["autoencoder"]
    model = ZonedFieldAutoencoder(
        num_zones=num_zones_from_geo(geo),
        latent_dim=int(a["latent_dim_per_zone"]),
        point_hidden=int(a["point_hidden"]),
        decoder_hidden=int(a["decoder_hidden"]),
        frequencies=int(a["coord_fourier_frequencies"]),
        zone_embedding_dim=int(a["zone_embedding_dim"]),
    ).to(device)
    load_checkpoint(
        output_path(cfg, "autoencoder", target, "model.pt"),
        model,
        cfg=cfg,
        map_location=device,
    )
    model.eval()
    return model


def make_sr(cfg, geo, method: str, device: torch.device):
    s = cfg["superresolution"]
    k = num_zones_from_geo(geo)
    condition = ConditionEncoder(
        int(cfg["model"]["history_length"]),
        int(s["condition_hidden"]),
        geo["device_zone"],
        k,
        int(s["condition_graph_blocks"]),
    )
    latent_dim = int(cfg["autoencoder"]["latent_dim_per_zone"])
    model = (
        DeterministicSR(condition, latent_dim)
        if method == "4dsrda"
        else DiffusionSR(condition, latent_dim, int(s["diffusion_hidden"]))
    )
    model = model.to(device)
    load_checkpoint(
        output_path(cfg, "superresolution", method, "model.pt"),
        model,
        cfg=cfg,
        map_location=device,
    )
    model.eval()
    return model


def condition_tensors(
    cfg,
    data,
    assimilation,
    frame: int,
    norm: LowDimNorm,
    device: torch.device,
    method: str,
):
    history = int(cfg["model"]["history_length"])
    pdiag = np.diagonal(assimilation["P_analysis"], axis1=1, axis2=2)
    p = norm.norm_pdiag(history_window(pdiag, frame, history))
    u = norm.norm_frame_u(history_window(data["frame_controls"], frame, history))
    d = norm.norm_innovation(history_window(assimilation["innovation"], frame, history))

    if method in STOCHASTIC_METHODS:
        if "analysis_ensemble" not in assimilation:
            raise KeyError("analysis_ensemble missing; rerun 01_zone_da.py --stage assimilate")
        ensemble = assimilation["analysis_ensemble"]
        configured_members = int(cfg["diffusion"]["ensemble_members"])
        enkf_members = int(cfg["enkf"]["ensemble_size"])
        if ensemble.shape[1] != enkf_members:
            raise ValueError(
                f"analysis_ensemble has {ensemble.shape[1]} members; expected {enkf_members}"
            )
        if configured_members != enkf_members:
            raise ValueError(
                "Complete analysis-ensemble propagation requires "
                "diffusion.ensemble_members == enkf.ensemble_size"
            )
        member_histories = np.stack(
            [history_window(ensemble[:, member, :], frame, history) for member in range(enkf_members)],
            axis=0,
        )
        arrays = (
            norm.norm_z(member_histories),
            np.broadcast_to(p[None], (enkf_members,) + p.shape).copy(),
            np.broadcast_to(u[None], (enkf_members,) + u.shape).copy(),
            np.broadcast_to(d[None], (enkf_members,) + d.shape).copy(),
        )
    else:
        arrays = (
            norm.norm_z(history_window(assimilation["z_analysis"], frame, history))[None],
            p[None],
            u[None],
            d[None],
        )
    return tuple(torch.as_tensor(x, dtype=torch.float32, device=device) for x in arrays)


@torch.no_grad()
def sample_latents(
    cfg,
    method: str,
    model,
    condition,
    h_zone,
    adjacency,
    mean,
    std,
    k: int,
    device: torch.device,
    generator=None,
):
    if method == "4dsrda":
        normalized = model(*condition, h_zone, adjacency)
    else:
        members = condition[0].shape[0]
        normalized = ddim_sample(
            model,
            (members, k, int(cfg["autoencoder"]["latent_dim_per_zone"])),
            (*condition, h_zone, adjacency),
            int(cfg["diffusion"]["sample_steps"]),
            int(cfg["diffusion"]["train_steps"]),
            float(cfg["diffusion"]["beta_start"]),
            float(cfg["diffusion"]["beta_end"]),
            device,
            generator=generator,
        )
    latent_mean = torch.as_tensor(mean, dtype=torch.float32, device=device)[None, None]
    latent_std = torch.as_tensor(std, dtype=torch.float32, device=device)[None, None]
    return normalized * latent_std + latent_mean


def stochastic_seed(cfg, method: str, sid: int, frame: int) -> int:
    method_offset = 600001 if method == "diffsrda" else 900001
    return int(cfg["project"]["seed"]) + method_offset + 1000003 * int(sid) + 1009 * int(frame)


def find_manifest_row(cfg: Mapping, sample_id: int) -> dict:
    for row in read_manifest(cfg):
        if int(row["sample_id"]) == int(sample_id):
            return row
    raise FileNotFoundError(f"sample {sample_id:06d} is absent from the prepared manifest")


def resolve_sample_id(cfg: Mapping, requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    path = output_path(cfg, "evaluation", "best_proposed_case.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run 05_evaluate.py first or provide --sample-id."
        )
    with path.open("r", encoding="utf-8") as f:
        return int(json.load(f)["sample_id"])


def load_model_bundle(cfg: Mapping, geo: Mapping, method: str, device: torch.device) -> ModelBundle:
    if method == "zone_const":
        return ModelBundle(method=method)
    target = target_name(method)
    ae = make_autoencoder(cfg, geo, target, device)
    sr = make_sr(cfg, geo, method, device)
    with np.load(output_path(cfg, "autoencoder", target, "latent_normalization_train_only.npz")) as z:
        latent_mean = z["mean"].astype(np.float32)
        latent_std = z["std"].astype(np.float32)
    with np.load(output_path(cfg, "autoencoder", target, "field_stats_train_only.npz")) as z:
        field_mean = float(z["mean"])
        field_std = float(z["std"])
    return ModelBundle(
        method=method,
        ae=ae,
        sr=sr,
        latent_mean=latent_mean,
        latent_std=latent_std,
        field_mean=field_mean,
        field_std=field_std,
    )


@torch.no_grad()
def decode_members(ae, latents, coords, zones, field_mean, field_std,
                   analysis_members=None):
    """Decode one cell chunk, optionally lifting corresponding EnKF members."""
    decoded = ae.decode(latents, coords, zones).cpu().numpy()
    decoded = decoded * field_std + field_mean
    if analysis_members is not None:
        # Proposed adds the residual directly; there is no recentering pass.
        decoded = decoded + zone_lift(analysis_members, zones.cpu().numpy() + 1)
    return decoded


@torch.no_grad()
def reconstruct_selected_cells(
    cfg: Mapping,
    geo: Mapping,
    data: Mapping,
    assimilation: Mapping,
    bundle: ModelBundle,
    frame: int,
    selected: np.ndarray,
    norm: LowDimNorm,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ensemble mean and standard deviation at selected CFD cells."""
    method = bundle.method
    zone_ids = np.asarray(geo["zone_id"][selected], dtype=np.int64)
    z_analysis = assimilation["z_analysis"][frame]
    if method == "zone_const":
        mean = zone_lift(z_analysis, zone_ids).astype(np.float32)
        return mean, np.zeros_like(mean)

    k = num_zones_from_geo(geo)
    h_zone = torch.as_tensor(geo["H_zone"], dtype=torch.float32, device=device)
    adjacency = torch.as_tensor(geo["adjacency"], dtype=torch.float32, device=device)
    coords = torch.as_tensor(geo["local_coords"][selected], dtype=torch.float32, device=device)
    zones = torch.as_tensor(zone_ids - 1, dtype=torch.long, device=device)

    condition = condition_tensors(cfg, data, assimilation, frame, norm, device, method)
    generator = None
    if method in STOCHASTIC_METHODS:
        generator = torch.Generator(device=device)
        generator.manual_seed(stochastic_seed(cfg, method, int(data["sample_id"]), frame))
    latents = sample_latents(
        cfg, method, bundle.sr, condition, h_zone, adjacency,
        bundle.latent_mean, bundle.latent_std, k, device, generator,
    )

    expected_members = int(cfg["diffusion"]["ensemble_members"]) if method in STOCHASTIC_METHODS else 1
    configured_chunk = int(cfg["reconstruct"]["decode_cell_chunk"])
    chunk = max(256, configured_chunk // max(expected_members, 1))

    if method == "proposed":
        analysis_members = assimilation["analysis_ensemble"][frame]
    else:
        analysis_members = None

    prediction_mean = np.empty(len(selected), dtype=np.float32)
    prediction_std = np.empty(len(selected), dtype=np.float32)
    for sl in iter_chunks(len(selected), chunk):
        members = decode_members(
            bundle.ae, latents, coords[sl], zones[sl],
            bundle.field_mean, bundle.field_std, analysis_members,
        )
        prediction_mean[sl] = members.mean(axis=0).astype(np.float32)
        prediction_std[sl] = members.std(axis=0, ddof=1 if members.shape[0] > 1 else 0).astype(np.float32)
    return prediction_mean, prediction_std
