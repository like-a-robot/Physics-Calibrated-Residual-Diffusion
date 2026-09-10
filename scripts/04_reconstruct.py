"""Reconstruct test trajectories and save deterministic and ensemble-reliability diagnostics."""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from common import (
    LowDimNorm, barrier, cli_common, configure_args, finish_runtime,
    independent_rows, iter_chunks, load_prepared, load_prepared_geometry,
    num_zones_from_geo, output_path, read_manifest, temperature_cache_path, zone_lift,
)
from inference import (
    METHODS, STOCHASTIC_METHODS, condition_tensors, decode_members,
    make_autoencoder, make_sr, sample_latents, stochastic_seed, target_name,
)


def complete_test_rows(cfg):
    rows = []
    for row in read_manifest(cfg):
        if row["split"] != "test":
            continue
        sid = int(row["sample_id"])
        if temperature_cache_path(cfg, sid).exists() and output_path(
            cfg, "zone_da", "assimilation", f"sample_{sid:06d}.npz"
        ).exists():
            rows.append(row)
    return rows


def uncertainty_levels(cfg) -> np.ndarray:
    values = np.asarray(cfg.get("uncertainty", {}).get("nominal_levels", [0.5, 0.8, 0.9]), dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("uncertainty.nominal_levels must contain values strictly between 0 and 1")
    return values


@torch.no_grad()
def reconstruct(cfg, rt, method: str):
    geo = load_prepared_geometry(cfg)
    norm = LowDimNorm.load(cfg)
    k = num_zones_from_geo(geo)
    rows = complete_test_rows(cfg)
    if not rows:
        raise RuntimeError("No test trajectories with assimilation outputs")

    coords = torch.as_tensor(geo["local_coords"], dtype=torch.float32, device=rt.device)
    zones = torch.as_tensor(
        geo["zone_id"].astype(np.int64) - 1,
        dtype=torch.long,
        device=rt.device,
    )
    h_zone = torch.as_tensor(geo["H_zone"], dtype=torch.float32, device=rt.device)
    adjacency = torch.as_tensor(geo["adjacency"], dtype=torch.float32, device=rt.device)
    volumes = np.asarray(geo["volumes"], dtype=np.float64)

    configured_chunk = int(cfg["reconstruct"]["decode_cell_chunk"])
    stochastic = method in STOCHASTIC_METHODS
    members_expected = int(cfg["diffusion"]["ensemble_members"]) if stochastic else 1
    # decode_cell_chunk is treated as an approximate member-cell budget.
    chunk = max(256, configured_chunk // max(1, members_expected))
    levels = uncertainty_levels(cfg)

    if method != "zone_const":
        target = target_name(method)
        ae = make_autoencoder(cfg, geo, target, rt.device)
        sr = make_sr(cfg, geo, method, rt.device)
        with np.load(output_path(cfg, "autoencoder", target, "latent_normalization_train_only.npz")) as z:
            latent_mean, latent_std = z["mean"], z["std"]
        with np.load(output_path(cfg, "autoencoder", target, "field_stats_train_only.npz")) as z:
            field_mean, field_std = float(z["mean"]), float(z["std"])

    sample_ids, full_curves, hot_curves = [], [], []
    coverage_volume_rows, total_volume_rows, rank_volume_rows = [], [], []
    times_s = load_prepared(cfg, int(rows[0]["sample_id"]))["times_s"]

    for row in independent_rows(rows, rt):
        sid = int(row["sample_id"])
        data = load_prepared(cfg, sid)
        times_s = data["times_s"]
        truth_all = np.load(temperature_cache_path(cfg, sid), mmap_mode="r")
        with np.load(output_path(cfg, "zone_da", "assimilation", f"sample_{sid:06d}.npz")) as z:
            assimilation = {name: z[name] for name in z.files}
        if stochastic and "analysis_ensemble" not in assimilation:
            raise KeyError(f"sample {sid:06d} missing analysis_ensemble; rerun assimilation")

        full = np.full(len(times_s), np.nan, dtype=np.float64)
        hot = np.full(len(times_s), np.nan, dtype=np.float64)
        coverage_volume = np.zeros(len(levels), dtype=np.float64)
        total_volume = 0.0
        rank_volume = np.zeros(members_expected + 1, dtype=np.float64)

        for frame in range(1, len(times_s)):
            truth = np.asarray(truth_all[frame], dtype=np.float32)
            hot_n = max(1, int(math.ceil(0.05 * len(truth))))
            hot_idx = np.argpartition(truth, len(truth) - hot_n)[-hot_n:]
            z_analysis = assimilation["z_analysis"][frame]

            if method == "zone_const":
                latents = None
                members_count = 1
                analysis_members = z_analysis[None]
            else:
                condition = condition_tensors(
                    cfg, data, assimilation, frame, norm, rt.device, method
                )
                generator = None
                if stochastic:
                    generator = torch.Generator(device=rt.device)
                    generator.manual_seed(stochastic_seed(cfg, method, sid, frame))
                latents = sample_latents(
                    cfg,
                    method,
                    sr,
                    condition,
                    h_zone,
                    adjacency,
                    latent_mean,
                    latent_std,
                    k,
                    rt.device,
                    generator=generator,
                )
                members_count = latents.shape[0]
                analysis_members = (
                    assimilation["analysis_ensemble"][frame]
                    if method == "proposed"
                    else None
                )

            sse = hot_sse = 0.0
            cell_count = 0
            quantile_probs = np.asarray(
                [p for q in levels for p in ((1.0 - q) / 2.0, (1.0 + q) / 2.0)],
                dtype=np.float64,
            )

            for sl in iter_chunks(len(truth), chunk):
                if method == "zone_const":
                    members = zone_lift(z_analysis, geo["zone_id"][sl])[None]
                else:
                    members = decode_members(
                        ae, latents, coords[sl], zones[sl], field_mean, field_std,
                        analysis_members,
                    )

                prediction = members.mean(axis=0)
                error = prediction.astype(np.float64) - truth[sl].astype(np.float64)
                sse += float(np.dot(error, error))
                cell_count += len(error)
                local_hot = hot_idx[(hot_idx >= sl.start) & (hot_idx < sl.stop)] - sl.start
                hot_sse += float(np.dot(error[local_hot], error[local_hot]))

                if stochastic:
                    local_truth = truth[sl]
                    local_volume = volumes[sl]
                    quantiles = np.quantile(members, quantile_probs, axis=0)
                    for level_index in range(len(levels)):
                        lower = quantiles[2 * level_index]
                        upper = quantiles[2 * level_index + 1]
                        inside = (local_truth >= lower) & (local_truth <= upper)
                        coverage_volume[level_index] += float(local_volume[inside].sum())
                    total_volume += float(local_volume.sum())

                    # Continuous temperatures make exact ties uncommon; rank is in [0, members_count].
                    ranks = np.sum(members < local_truth[None, :], axis=0).astype(np.int64)
                    rank_volume += np.bincount(
                        ranks,
                        weights=local_volume,
                        minlength=members_count + 1,
                    )

            full[frame] = math.sqrt(sse / cell_count)
            hot[frame] = math.sqrt(hot_sse / hot_n)

        sample_ids.append(sid)
        full_curves.append(full)
        hot_curves.append(hot)
        coverage_volume_rows.append(coverage_volume)
        total_volume_rows.append(total_volume)
        rank_volume_rows.append(rank_volume)
        print(f"reconstructed {method} sample {sid:06d}")

    n_frames = len(times_s)
    full_array = np.stack(full_curves) if full_curves else np.empty((0, n_frames), dtype=np.float64)
    hot_array = np.stack(hot_curves) if hot_curves else np.empty((0, n_frames), dtype=np.float64)
    coverage_array = (
        np.stack(coverage_volume_rows)
        if coverage_volume_rows
        else np.empty((0, len(levels)), dtype=np.float64)
    )
    rank_array = (
        np.stack(rank_volume_rows)
        if rank_volume_rows
        else np.empty((0, members_expected + 1), dtype=np.float64)
    )
    shard = output_path(cfg, "reconstruction", method, "shards", mkdir=True) / f"rank_{rt.rank:04d}.npz"
    np.savez(
        shard,
        sample_ids=np.asarray(sample_ids, dtype=np.int32),
        full_rmse=full_array,
        hotspot_rmse=hot_array,
        nominal_levels=levels.astype(np.float64),
        coverage_volume=coverage_array,
        total_volume=np.asarray(total_volume_rows, dtype=np.float64),
        rank_volume=rank_array,
        times_s=np.asarray(times_s, dtype=np.int32),
    )
    barrier(rt)


def main(args):
    cfg, rt = configure_args(args)
    try:
        reconstruct(cfg, rt, args.method)
    finally:
        finish_runtime(rt)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    cli_common(parser)
    parser.add_argument("--method", choices=METHODS, required=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
