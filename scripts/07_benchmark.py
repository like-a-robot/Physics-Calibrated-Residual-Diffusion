"""Benchmark online zone assimilation and full-field reconstruction latency."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time

import numpy as np
import torch

from common import (
    LowDimNorm,
    LowParameterZoneDynamics,
    cli_common,
    configure_args,
    finish_runtime,
    history_window,
    load_checkpoint,
    load_prepared,
    load_prepared_geometry,
    num_zones_from_geo,
    output_path,
)

from inference import (
    DISPLAY, METHODS, load_model_bundle, reconstruct_selected_cells, resolve_sample_id,
)


def make_zone_model(cfg, geo, norm, device: torch.device):
    d = cfg["dynamics"]
    model = LowParameterZoneDynamics(
        geo=geo,
        norm=norm,
        hidden=int(d["residual_hidden"]),
        tcn_blocks=int(d["residual_tcn_blocks"]),
        graph_blocks=int(d["residual_graph_blocks"]),
        residual_max_C=float(d["residual_max_C_per_step"]),
        initial_gains_C=d["initial_core_gains_C"],
    ).to(device)
    load_checkpoint(output_path(cfg, "zone_da", "model.pt"), model, cfg=cfg, map_location=device)
    model.eval()
    return model


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(fn, device: torch.device) -> tuple[float, float]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    fn()
    synchronize(device)
    elapsed = time.perf_counter() - start
    peak_mb = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
        if device.type == "cuda"
        else float("nan")
    )
    return elapsed, peak_mb


@torch.no_grad()
def prepare_da_step(cfg, geo, data, assimilation, frame: int, norm, model, device):
    """Prepare a deterministic one-step EnKF benchmark from the saved previous analysis ensemble."""
    if frame <= 0:
        raise ValueError("Benchmark frames must be greater than zero")
    members = int(cfg["enkf"]["ensemble_size"])
    history_length = int(cfg["model"]["history_length"])
    analysis_ensemble = np.asarray(assimilation["analysis_ensemble"], dtype=np.float64)
    if analysis_ensemble.shape[1] != members:
        raise ValueError("Saved analysis ensemble size differs from config")

    history = np.stack(
        [history_window(analysis_ensemble[:, member, :], frame - 1, history_length) for member in range(members)],
        axis=0,
    )
    u_hist = history_window(data["transition_controls"], frame - 1, history_length)
    controls = np.repeat(u_hist[None], members, axis=0)
    observation = np.asarray(data["sensor_y"][frame], dtype=np.float64)

    with np.load(output_path(cfg, "zone_da", "noise_model.npz")) as z:
        q = np.asarray(z["Q"], dtype=np.float64)
        rmat = np.asarray(z["R"], dtype=np.float64)
        bias = np.asarray(z["b"], dtype=np.float64)
    hmat = np.asarray(geo["H_zone"], dtype=np.float64)

    rng = np.random.default_rng(int(cfg["project"]["seed"]) + 700001 + frame)
    process_noise = rng.multivariate_normal(np.zeros(q.shape[0]), q, size=members)
    observation_noise = rng.multivariate_normal(np.zeros(rmat.shape[0]), rmat, size=members)
    z_tensor = torch.as_tensor(history, dtype=torch.float32, device=device)
    u_tensor = torch.as_tensor(controls, dtype=torch.float32, device=device)

    def step():
        with torch.no_grad():
            pred = model(z_tensor, u_tensor).cpu().numpy().astype(np.float64)
        ensemble_b = pred + process_noise
        predicted_obs = ensemble_b @ hmat.T + bias
        pb = np.cov(ensemble_b, rowvar=False)
        s = hmat @ pb @ hmat.T + rmat
        gain = np.linalg.solve(s, hmat @ pb).T
        perturbed = observation + observation_noise
        _ = ensemble_b + (perturbed - predicted_obs) @ gain.T

    return step


def select_frames(cfg, data, all_frames: bool, requested_times: list[int] | None) -> list[int]:
    if all_frames:
        return list(range(1, len(data["times_s"])))
    if requested_times:
        time_to_frame = {int(t): i for i, t in enumerate(data["times_s"])}
        missing = [t for t in requested_times if int(t) not in time_to_frame]
        if missing:
            raise ValueError(f"Benchmark times are absent from the trajectory: {missing}")
        frames = [time_to_frame[int(t)] for t in requested_times]
    else:
        configured = cfg.get("benchmark", {}).get("times_s", [600, 1200, 1800])
        time_to_frame = {int(t): i for i, t in enumerate(data["times_s"])}
        frames = [time_to_frame[int(t)] for t in configured if int(t) in time_to_frame]
        if not frames and len(data["times_s"]) > 1:
            frames = list(range(1, min(len(data["times_s"]), 4)))
    frames = sorted(set(int(x) for x in frames if int(x) > 0))
    if not frames:
        raise ValueError("No valid benchmark frames were selected")
    return frames


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1) if len(array) > 1 else 0.0)


def benchmark(args) -> None:
    cfg, rt = configure_args(args)
    try:
        if not rt.main:
            return
        sid = resolve_sample_id(cfg, args.sample_id)
        geo = load_prepared_geometry(cfg)
        data = load_prepared(cfg, sid)
        norm = LowDimNorm.load(cfg)
        with np.load(output_path(cfg, "zone_da", "assimilation", f"sample_{sid:06d}.npz")) as z:
            assimilation = {name: z[name] for name in z.files}

        frames = select_frames(cfg, data, args.all_frames, args.times)
        repeats = 1 if args.all_frames else int(cfg.get("benchmark", {}).get("repeats", 3))
        warmup = int(cfg.get("benchmark", {}).get("warmup", 1))
        update_interval_s = float(cfg.get("benchmark", {}).get("update_interval_s", 30.0))
        trajectory_steps = len(data["times_s"]) - 1

        zone_model = make_zone_model(cfg, geo, norm, rt.device)
        da_steps = {
            frame: prepare_da_step(cfg, geo, data, assimilation, frame, norm, zone_model, rt.device)
            for frame in frames
        }
        for frame in frames:
            for _ in range(warmup):
                da_steps[frame]()
        da_by_frame: dict[int, list[float]] = {frame: [] for frame in frames}
        da_peak = 0.0
        for frame in frames:
            for _ in range(repeats):
                elapsed, peak = timed_call(da_steps[frame], rt.device)
                da_by_frame[frame].append(elapsed)
                if np.isfinite(peak):
                    da_peak = max(da_peak, peak)
        da_flat = [value for frame in frames for value in da_by_frame[frame]]
        da_mean, da_std = mean_std(da_flat)

        selected = np.arange(len(geo["zone_id"]), dtype=np.int64)
        hardware = (
            torch.cuda.get_device_name(rt.device)
            if rt.device.type == "cuda"
            else (platform.processor() or platform.machine())
        )
        rows = []
        for method in METHODS:
            bundle = load_model_bundle(cfg, geo, method, rt.device)

            def reconstruction_step(frame: int):
                return lambda: reconstruct_selected_cells(
                    cfg,
                    geo,
                    data,
                    assimilation,
                    bundle,
                    frame,
                    selected,
                    norm,
                    rt.device,
                )

            steps = {frame: reconstruction_step(frame) for frame in frames}
            for frame in frames:
                for _ in range(warmup):
                    steps[frame]()

            reconstruction_by_frame: dict[int, list[float]] = {frame: [] for frame in frames}
            reconstruction_peak = 0.0
            for frame in frames:
                for _ in range(repeats):
                    elapsed, peak = timed_call(steps[frame], rt.device)
                    reconstruction_by_frame[frame].append(elapsed)
                    if np.isfinite(peak):
                        reconstruction_peak = max(reconstruction_peak, peak)

            reconstruction_flat = [
                value for frame in frames for value in reconstruction_by_frame[frame]
            ]
            rec_mean, rec_std = mean_std(reconstruction_flat)
            total_mean = da_mean + rec_mean
            total_std = float(np.sqrt(da_std ** 2 + rec_std ** 2))

            if args.all_frames:
                trajectory_runtime = sum(
                    float(np.mean(da_by_frame[frame]))
                    + float(np.mean(reconstruction_by_frame[frame]))
                    for frame in frames
                )
                trajectory_type = "measured"
            else:
                trajectory_runtime = total_mean * trajectory_steps
                trajectory_type = "estimated_from_sampled_frames"

            rows.append({
                "method": DISPLAY[method],
                "sample_id": sid,
                "frames_benchmarked": len(frames),
                "repeats_per_frame": repeats,
                "zone_DA_latency_mean_s": da_mean,
                "zone_DA_latency_std_s": da_std,
                "reconstruction_latency_mean_s": rec_mean,
                "reconstruction_latency_std_s": rec_std,
                "total_online_latency_mean_s": total_mean,
                "total_online_latency_std_s": total_std,
                "real_time_factor": total_mean / update_interval_s,
                "trajectory_runtime_s": trajectory_runtime,
                "trajectory_runtime_type": trajectory_type,
                "peak_GPU_memory_MB": max(da_peak, reconstruction_peak) if rt.device.type == "cuda" else None,
                "device": str(rt.device),
                "hardware": hardware,
                "torch_version": torch.__version__,
                "diffusion_steps": int(cfg["diffusion"]["sample_steps"]) if method in {"diffsrda", "proposed"} else 0,
                "ensemble_members": int(cfg["diffusion"]["ensemble_members"]) if method in {"diffsrda", "proposed"} else 1,
            })
            print(json.dumps(rows[-1], ensure_ascii=False, indent=2))

            del bundle
            if rt.device.type == "cuda":
                torch.cuda.empty_cache()

        eval_dir = output_path(cfg, "evaluation", mkdir=True)
        csv_path = eval_dir / "computational_efficiency.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (eval_dir / "computational_efficiency.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"saved benchmark results to {csv_path}")
    finally:
        finish_runtime(rt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    cli_common(parser)
    parser.add_argument("--sample-id", type=int, default=None)
    parser.add_argument("--times", type=int, nargs="+", default=None)
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Benchmark all 60 online updates once and report a measured trajectory runtime.",
    )
    return parser


if __name__ == "__main__":
    benchmark(build_parser().parse_args())
