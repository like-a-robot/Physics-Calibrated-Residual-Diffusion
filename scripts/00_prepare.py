"""Validate raw data and build the unified connected-zone dataset."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import (
    CONTROL_DIM, atomic_json, cli_common, configure_args, control_sequences,
    device_zone_mapping, extract_npz_array, finish_runtime, load_geometry,
    num_zones_from_geo, output_path, physics_features_np, prepared_sample_path,
    save_geometry, scan_manifest, sensor_mapping, temperature_cache_path,
    write_manifest, zone_reduce,
)


def select_smoke_rows(rows, maximum: int):
    selected, used = [], set()
    for split in ("train", "validation", "test"):
        row = next((x for x in rows if x["split"] == split), None)
        if row is not None:
            selected.append(row); used.add(int(row["sample_id"]))
    for row in rows:
        if len(selected) >= maximum:
            break
        if int(row["sample_id"]) not in used:
            selected.append(row)
    return selected[:maximum]


def expected_times(cfg):
    spec = cfg["data"]["expected_times_s"]
    return np.arange(int(spec["start"]), int(spec["stop"]) + int(spec["step"]), int(spec["step"]), dtype=np.int64)


def safe_std(x, axis=0):
    return np.maximum(np.std(x, axis=axis), 1e-6).astype(np.float32)


def prepare(args):
    cfg, rt = configure_args(args)
    try:
        if not rt.main:
            return
        root = output_path(cfg, "prepared", mkdir=True)
        rows, split = scan_manifest(cfg)
        write_manifest(root / "manifest.csv", rows)
        atomic_json(root / "split.json", {
            "seed": int(cfg["project"]["seed"]),
            "fractions": {k: float(cfg["split"][k]) for k in ("train", "validation", "test")},
            "sample_ids": split,
        })

        geo = load_geometry(cfg, smoke=args.smoke)
        k = num_zones_from_geo(geo)
        summary_path = cfg["data"].get("zone_summary_json")
        if summary_path and Path(summary_path).exists() and not args.smoke:
            with open(summary_path, "r", encoding="utf-8-sig") as f:
                zone_summary = json.load(f)
            if int(zone_summary.get("cell_count", len(geo["zone_id"]))) != int(cfg["data"]["expected_cells"]):
                raise ValueError("Connected-zone summary cell_count does not match config")
            if int(zone_summary.get("nonzero_fluid_zone_count", k)) != k:
                raise ValueError("Connected-zone summary nonzero_fluid_zone_count does not match active zones")
        device_map, equipment_rows = device_zone_mapping(
            cfg, geo["centers"], geo["zone_id"], geo["volumes"], k
        )
        sensor_local, sensor_zone, h_zone, sensor_rows = sensor_mapping(
            cfg, geo["centers"], geo["zone_id"], k
        )
        sensor_xyz = np.asarray(
            [[float(row["x_m"]), float(row["y_m"]), float(row["z_m"])] for row in sensor_rows],
            dtype=np.float32,
        )
        geo.update({
            "device_zone": device_map.astype(np.float32),
            "sensor_cell_local": sensor_local.astype(np.int64),
            "sensor_cell_index": geo["cell_indices"][sensor_local].astype(np.int64),
            "sensor_zone_id": sensor_zone.astype(np.int16),
            "sensor_xyz": sensor_xyz,
            "H_zone": h_zone.astype(np.float32),
        })
        save_geometry(root / "geometry.npz", geo)
        atomic_json(root / "equipment_order.json", equipment_rows)
        atomic_json(root / "sensor_mapping.json", [
            {**row, "sensor_cell_index": int(geo["cell_indices"][sensor_local[i]]),
             "sensor_zone_id": int(sensor_zone[i])}
            for i, row in enumerate(sensor_rows)
        ])

        work_rows = select_smoke_rows(rows, int(cfg["smoke"]["max_samples"])) if args.smoke else rows
        max_frames = int(cfg["smoke"]["max_frames"]) if args.smoke else int(cfg["data"]["expected_frames"])
        wanted_times = expected_times(cfg)
        cache_dtype = np.dtype(cfg["data"].get("cache_temperature_dtype", "float32"))
        summaries = []

        for row in work_rows:
            sid = int(row["sample_id"]); source = Path(row["sample_path"])
            cache = temperature_cache_path(cfg, sid)
            if not (args.resume and cache.exists()):
                extract_npz_array(
                    source, cfg["data"]["temperature_key"], cache,
                    selected_columns=geo["cell_indices"] if args.smoke else None,
                    output_dtype=cache_dtype,
                )
            temperature = np.load(cache, mmap_mode="r")
            with np.load(source, allow_pickle=False) as raw:
                times = np.asarray(raw[cfg["data"]["time_key"]], dtype=np.int64)
            if not np.array_equal(times, wanted_times):
                raise ValueError(f"{source} time grid differs from expected_times_s")
            frames = min(max_frames, len(times))
            z_true = np.empty((frames, k), dtype=np.float32)
            sensor_y = np.empty((frames, len(sensor_local)), dtype=np.float32)
            t_min, t_max = math.inf, -math.inf
            for frame in range(frames):
                field = np.asarray(temperature[frame], dtype=np.float32)
                if not np.isfinite(field).all():
                    raise ValueError(f"NaN/Inf in sample {sid}, frame {frame}")
                z_true[frame] = zone_reduce(field, geo["zone_id"], geo["volumes"], k)
                sensor_y[frame] = field[sensor_local]
                t_min = min(t_min, float(field.min())); t_max = max(t_max, float(field.max()))
            frame_u, transition_u = control_sequences(row, frames)
            assert frame_u.shape == (frames, CONTROL_DIM)
            assert transition_u.shape == (frames - 1, CONTROL_DIM)
            dst = prepared_sample_path(cfg, sid); dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                dst, sample_id=np.int32(sid), times_s=times[:frames].astype(np.int32),
                z_true=z_true, sensor_y=sensor_y,
                frame_controls=frame_u, transition_controls=transition_u,
            )
            summaries.append({"sample_id": sid, "split": row["split"], "frames": frames,
                              "temperature_min_C": t_min, "temperature_max_C": t_max})
            print(f"prepared sample {sid:06d}: frames={frames}, cells={temperature.shape[1]}, zones={k}")

        train_ids = [int(r["sample_id"]) for r in work_rows if r["split"] == "train"]
        if not train_ids:
            raise RuntimeError("No prepared training cases")
        loaded = [np.load(prepared_sample_path(cfg, sid), allow_pickle=False) for sid in train_ids]
        try:
            z = np.concatenate([x["z_true"] for x in loaded], axis=0)
            frame_u = np.concatenate([x["frame_controls"] for x in loaded], axis=0)
            transition_u = np.concatenate([x["transition_controls"] for x in loaded], axis=0)
            y = np.concatenate([x["sensor_y"] for x in loaded], axis=0)
            tf, rf, cf = [], [], []
            for x in loaded:
                if len(x["transition_controls"]) == 0:
                    continue
                t, r, c = physics_features_np(
                    x["z_true"][:-1], x["transition_controls"],
                    geo["exchange_weights"], geo["device_zone"], geo["zone_volumes"]
                )
                tf.append(t); rf.append(r); cf.append(c)
            physics_scales = np.array([
                np.sqrt(np.mean(np.concatenate(tf) ** 2)),
                np.sqrt(np.mean(np.concatenate(rf) ** 2)),
                np.sqrt(np.mean(np.concatenate(cf) ** 2)),
            ], dtype=np.float32)
            physics_scales = np.maximum(physics_scales, 1e-6)
        finally:
            for x in loaded:
                x.close()
        np.savez(
            root / "normalization_train_only.npz",
            z_mean=z.mean(axis=0).astype(np.float32), z_std=safe_std(z),
            u_frame_mean=frame_u.mean(axis=0).astype(np.float32), u_frame_std=safe_std(frame_u),
            u_transition_mean=transition_u.mean(axis=0).astype(np.float32), u_transition_std=safe_std(transition_u),
            y_mean=y.mean(axis=0).astype(np.float32), y_std=safe_std(y),
            physics_scales=physics_scales,
        )

        test_z = np.linspace(18.0, 30.0, k, dtype=np.float32)
        lifted = test_z[geo["zone_id"].astype(np.int64) - 1]
        reduced = zone_reduce(lifted, geo["zone_id"], geo["volumes"], k)
        report = {
            "actual_sample_count": len(rows), "prepared_sample_count": len(work_rows),
            "selected_cells": int(len(geo["zone_id"])), "num_fluid_zones": k,
            "active_zone_ids": list(range(1, k + 1)),
            "equipment": {"Rack": 70, "CRAC": 46}, "sensors": 12,
            "graph_source": str(np.asarray(geo["graph_source"]).item()),
            "GU_max_abs_error": float(np.max(np.abs(reduced - test_z))),
            "trajectory_summaries": summaries,
        }
        atomic_json(root / "data_audit.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        finish_runtime(rt)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__); cli_common(p); return p


if __name__ == "__main__":
    prepare(build_parser().parse_args())
