"""Train the low-parameter connected-zone dynamics and run 40-D EnKF."""
from __future__ import annotations

import argparse
import csv
import math

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    LowDimNorm, LowParameterZoneDynamics, SENSOR_COUNT, autocast_context, barrier,
    broadcast_bool, cli_common, configure_args, distributed_mean, epoch_batches,
    finish_runtime, history_window, independent_rows, load_checkpoint, load_prepared,
    load_prepared_geometry, make_scaler, maybe_ddp, num_zones_from_geo, output_path,
    read_manifest, save_checkpoint,
)


def available_rows(cfg, split=None):
    rows = []
    for row in read_manifest(cfg):
        if split is not None and row["split"] != split:
            continue
        if output_path(cfg, "prepared", "trajectories", f"sample_{int(row['sample_id']):06d}.npz").exists():
            rows.append(row)
    return rows


def make_model(cfg, geo, norm):
    d = cfg["dynamics"]
    return LowParameterZoneDynamics(
        geo=geo, norm=norm, hidden=int(d["residual_hidden"]),
        tcn_blocks=int(d["residual_tcn_blocks"]),
        graph_blocks=int(d["residual_graph_blocks"]),
        residual_max_C=float(d["residual_max_C_per_step"]),
        initial_gains_C=d["initial_core_gains_C"],
    )


def load_cases(cfg, rows):
    return {int(r["sample_id"]): load_prepared(cfg, int(r["sample_id"])) for r in rows}


def rollout_examples(cases, rollout):
    return [(sid, t) for sid, data in cases.items() for t in range(max(0, len(data["z_true"]) - int(rollout)))]


def make_initial_batch(batch, cases, history, device):
    z = np.stack([history_window(cases[sid]["z_true"], t, history) for sid, t in batch])
    return torch.as_tensor(z, dtype=torch.float32, device=device)


def controls_batch(batch, cases, end_offset, history, device):
    u = np.stack([
        history_window(cases[sid]["transition_controls"], t + end_offset, history)
        for sid, t in batch
    ])
    return torch.as_tensor(u, dtype=torch.float32, device=device)


def target_batch(batch, cases, step, device):
    x = np.stack([cases[sid]["z_true"][t + step + 1] for sid, t in batch])
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def validate(cfg, rt, model, cases, norm):
    model.eval(); history = int(cfg["model"]["history_length"])
    items = [(sid, t) for sid, data in cases.items() for t in range(len(data["z_true"]) - 1)]
    total = count = 0
    max_batches = int(cfg["train"]["validation_batches"])
    zstd = torch.as_tensor(norm.z_std, dtype=torch.float32, device=rt.device)
    with torch.no_grad():
        for bi, batch in enumerate(epoch_batches(items, int(cfg["train"]["zone_batch_size"]), int(cfg["project"]["seed"]), rt)):
            if max_batches > 0 and bi >= max_batches:
                break
            z = make_initial_batch(batch, cases, history, rt.device)
            u = controls_batch(batch, cases, 0, history, rt.device)
            target = target_batch(batch, cases, 0, rt.device)
            pred = model(z, u)
            loss = (((pred - target) / zstd) ** 2).sum()
            total += float(loss); count += target.numel()
    return distributed_mean(total, count, rt)


def append_log(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def train(cfg, rt, resume=False):
    geo = load_prepared_geometry(cfg); norm = LowDimNorm.load(cfg)
    model = maybe_ddp(make_model(cfg, geo, norm), rt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["learning_rate"]),
                                  weight_decay=float(cfg["train"]["weight_decay"]))
    scaler = make_scaler(rt.device, bool(cfg["train"]["amp"]))
    checkpoint = output_path(cfg, "zone_da", "model.pt")
    start_epoch, best = 0, math.inf
    if resume and checkpoint.exists():
        state = load_checkpoint(checkpoint, model, optimizer, scaler, cfg, map_location=rt.device)
        start_epoch, best = int(state["epoch"]) + 1, float(state["best"])
    train_rows = available_rows(cfg, "train"); val_rows = available_rows(cfg, "validation") or train_rows
    if not train_rows:
        raise RuntimeError("No prepared training trajectories")
    train_cases, val_cases = load_cases(cfg, train_rows), load_cases(cfg, val_rows)
    rollout = int(cfg["model"]["rollout_steps"]); history = int(cfg["model"]["history_length"])
    items = rollout_examples(train_cases, rollout)
    patience = 0; zstd = torch.as_tensor(norm.z_std, dtype=torch.float32, device=rt.device)
    max_smoke_batches = int(cfg["smoke"]["batches"]) if cfg["_smoke"] else 0
    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
        model.train(); total = count = 0
        for bi, batch in enumerate(epoch_batches(items, int(cfg["train"]["zone_batch_size"]), int(cfg["project"]["seed"]) + epoch, rt)):
            if max_smoke_batches and bi >= max_smoke_batches:
                break
            current = make_initial_batch(batch, train_cases, history, rt.device)
            optimizer.zero_grad(set_to_none=True); losses = []; residual_penalties = []
            with autocast_context(rt.device, bool(cfg["train"]["amp"])):
                for step in range(rollout):
                    controls = controls_batch(batch, train_cases, step, history, rt.device)
                    target = target_batch(batch, train_cases, step, rt.device)
                    pred, terms = model(current, controls, return_terms=True)
                    losses.append((((pred - target) / zstd) ** 2).mean())
                    residual_penalties.append((terms["residual"] / zstd).pow(2).mean())
                    current = torch.cat([current[:, 1:], pred[:, None]], dim=1)
                loss = losses[0]
                if len(losses) > 1:
                    loss = loss + float(cfg["model"]["rollout_weight"]) * torch.stack(losses[1:]).mean()
                loss = loss + float(cfg["dynamics"]["residual_l2_weight"]) * torch.stack(residual_penalties).mean()
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            scaler.step(optimizer); scaler.update()
            total += float(loss.detach()); count += 1
        train_loss = distributed_mean(total, count, rt); val_loss = validate(cfg, rt, model, val_cases, norm)
        stop = False
        if rt.main:
            beta = F.softplus((model.module if hasattr(model, "module") else model).raw_beta).detach().cpu().numpy()
            row = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                   "beta_transport_C": float(beta[0]), "beta_rack_C": float(beta[1]), "beta_crac_C": float(beta[2])}
            append_log(output_path(cfg, "zone_da", "training_log.csv"), row)
            print(row)
            if val_loss < best:
                best, patience = val_loss, 0
                save_checkpoint(checkpoint, model, optimizer, scaler, epoch, best, cfg,
                                extra={"num_zones": num_zones_from_geo(geo), "gains_C": beta.tolist()})
            else:
                patience += 1
            stop = patience >= int(cfg["train"]["patience"])
        stop = broadcast_bool(stop, rt); barrier(rt)
        if stop:
            break


@torch.no_grad()
def model_predict(model, z_history, u_history, device, return_terms=False):
    z = torch.as_tensor(z_history, dtype=torch.float32, device=device)
    u = torch.as_tensor(u_history, dtype=torch.float32, device=device)
    if return_terms:
        prediction, terms = model(z, u, return_terms=True)
        arrays = {key: value.cpu().numpy() for key, value in terms.items() if key != "beta"}
        arrays["beta"] = terms["beta"].cpu().numpy()
        return prediction.cpu().numpy(), arrays
    return model(z, u).cpu().numpy()


def nearest_psd(matrix, floor):
    x = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T)
    values, vectors = np.linalg.eigh(x); values = np.maximum(values, float(floor))
    return (vectors * values) @ vectors.T


@torch.no_grad()
def fit_statistics(cfg, rt):
    if not rt.main:
        return
    geo = load_prepared_geometry(cfg); norm = LowDimNorm.load(cfg); k = num_zones_from_geo(geo)
    model = make_model(cfg, geo, norm).to(rt.device)
    load_checkpoint(output_path(cfg, "zone_da", "model.pt"), model, cfg=cfg, map_location=rt.device); model.eval()
    history = int(cfg["model"]["history_length"]); residuals = []; observation_errors = []
    for row in available_rows(cfg, "train"):
        data = load_prepared(cfg, int(row["sample_id"]))
        observation_errors.append(data["sensor_y"] - data["z_true"] @ geo["H_zone"].T)
        for t in range(len(data["z_true"]) - 1):
            zh = history_window(data["z_true"], t, history)[None]
            uh = history_window(data["transition_controls"], t, history)[None]
            residuals.append(data["z_true"][t + 1] - model_predict(model, zh, uh, rt.device)[0])
    e = np.asarray(residuals, dtype=np.float64); o = np.concatenate(observation_errors).astype(np.float64)
    bias = o.mean(axis=0); centered = o - bias
    q = np.cov(e, rowvar=False) if len(e) > 1 else np.diag(np.maximum(e[0] ** 2, 1e-6))
    r = np.cov(centered, rowvar=False) if len(centered) > 1 else np.diag(np.maximum(centered[0] ** 2, 1e-6))
    q += np.eye(k) * float(cfg["enkf"]["process_jitter"])
    r += np.eye(SENSOR_COUNT) * float(cfg["enkf"]["observation_jitter"])
    floor = float(cfg["enkf"]["covariance_eigenvalue_floor"])
    q, r = nearest_psd(q, floor), nearest_psd(r, floor)
    path = output_path(cfg, "zone_da", "noise_model.npz"); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, Q=q.astype(np.float32), R=r.astype(np.float32), b=bias.astype(np.float32))
    print(f"saved Q/R/b to {path}")


@torch.no_grad()
def assimilate(cfg, rt):
    geo = load_prepared_geometry(cfg); norm = LowDimNorm.load(cfg); k = num_zones_from_geo(geo)
    with np.load(output_path(cfg, "zone_da", "noise_model.npz")) as s:
        q, rmat, bias = s["Q"].astype(np.float64), s["R"].astype(np.float64), s["b"].astype(np.float64)
    hmat = geo["H_zone"].astype(np.float64)
    model = make_model(cfg, geo, norm).to(rt.device)
    load_checkpoint(output_path(cfg, "zone_da", "model.pt"), model, cfg=cfg, map_location=rt.device); model.eval()
    members = int(cfg["enkf"]["ensemble_size"]); history_length = int(cfg["model"]["history_length"])
    out_dir = output_path(cfg, "zone_da", "assimilation", mkdir=True)
    for row in independent_rows(available_rows(cfg), rt):
        sid = int(row["sample_id"]); data = load_prepared(cfg, sid)
        rng = np.random.default_rng(int(cfg["project"]["seed"]) + 1009 * sid)
        frames = len(data["z_true"]); ensemble = np.repeat(data["z_true"][0][None], members, axis=0).astype(np.float64)
        history = np.repeat(ensemble[:, None, :], history_length, axis=1)
        forecast = np.empty((frames, k), dtype=np.float32); analysis = np.empty_like(forecast)
        covariance = np.empty((frames, k, k), dtype=np.float32)
        analysis_ensemble = np.empty((frames, members, k), dtype=np.float32)
        innovation = np.zeros((frames, SENSOR_COUNT), dtype=np.float32)
        term_transport = np.zeros((frames, k), dtype=np.float32)
        term_rack = np.zeros((frames, k), dtype=np.float32)
        term_crac = np.zeros((frames, k), dtype=np.float32)
        term_residual = np.zeros((frames, k), dtype=np.float32)
        forecast[0] = analysis[0] = ensemble.mean(axis=0); covariance[0] = 0.0
        analysis_ensemble[0] = ensemble.astype(np.float32)
        for t in range(frames - 1):
            u_hist = history_window(data["transition_controls"], t, history_length)
            controls = np.repeat(u_hist[None], members, axis=0)
            pred, terms = model_predict(model, history, controls, rt.device, return_terms=True)
            term_transport[t + 1] = terms["transport"].mean(axis=0)
            term_rack[t + 1] = terms["rack"].mean(axis=0)
            term_crac[t + 1] = terms["crac"].mean(axis=0)
            term_residual[t + 1] = terms["residual"].mean(axis=0)
            ensemble_b = pred.astype(np.float64) + rng.multivariate_normal(np.zeros(k), q, size=members)
            forecast[t + 1] = ensemble_b.mean(axis=0)
            observation = data["sensor_y"][t + 1].astype(np.float64)
            predicted_obs = ensemble_b @ hmat.T + bias
            innovation[t + 1] = observation - predicted_obs.mean(axis=0)
            pb = np.cov(ensemble_b, rowvar=False); s = hmat @ pb @ hmat.T + rmat
            gain = np.linalg.solve(s, hmat @ pb).T
            perturbed = observation + rng.multivariate_normal(np.zeros(SENSOR_COUNT), rmat, size=members)
            ensemble = ensemble_b + (perturbed - predicted_obs) @ gain.T
            analysis[t + 1] = ensemble.mean(axis=0); covariance[t + 1] = np.cov(ensemble, rowvar=False)
            analysis_ensemble[t + 1] = ensemble.astype(np.float32)
            history = np.concatenate([history[:, 1:], ensemble[:, None]], axis=1)
        np.savez_compressed(
            out_dir / f"sample_{sid:06d}.npz",
            z_forecast=forecast, z_analysis=analysis, P_analysis=covariance,
            analysis_ensemble=analysis_ensemble,
            innovation=innovation, times_s=data["times_s"],
            term_transport_C=term_transport, term_rack_C=term_rack,
            term_crac_C=term_crac, term_residual_C=term_residual,
        )
        print(f"assimilated sample {sid:06d}")


def main(args):
    cfg, rt = configure_args(args)
    try:
        stages = ["train", "fit_statistics", "assimilate"] if args.stage == "all" else [args.stage]
        for stage in stages:
            {"train": lambda: train(cfg, rt, args.resume),
             "fit_statistics": lambda: fit_statistics(cfg, rt),
             "assimilate": lambda: assimilate(cfg, rt)}[stage]()
            barrier(rt)
    finally:
        finish_runtime(rt)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__); cli_common(p)
    p.add_argument("--stage", choices=["train", "fit_statistics", "assimilate", "all"], default="all")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
