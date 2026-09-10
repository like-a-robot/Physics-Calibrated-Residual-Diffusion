"""Train deterministic full-latent and ensemble-conditioned diffusion SR models."""
from __future__ import annotations

import argparse
import csv
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    ConditionEncoder, DeterministicSR, DiffusionSR, LowDimNorm, autocast_context,
    barrier, broadcast_bool, cli_common, configure_args, diffusion_schedule,
    distributed_mean, epoch_batches, finish_runtime, history_window, load_checkpoint,
    load_prepared, load_prepared_geometry, make_scaler, maybe_ddp,
    num_zones_from_geo, output_path, read_manifest, save_checkpoint,
)

STOCHASTIC_METHODS = {"diffsrda", "proposed"}


def target_name(method: str) -> str:
    return "residual" if method == "proposed" else "full"


def latent_path(cfg, target: str, sid: int):
    return output_path(cfg, "autoencoder", target, "latents", f"sample_{sid:06d}.npz")


def available_rows(cfg, method: str, split: Optional[str] = None):
    target = target_name(method)
    rows = []
    for row in read_manifest(cfg):
        if split is not None and row["split"] != split:
            continue
        sid = int(row["sample_id"])
        if latent_path(cfg, target, sid).exists() and output_path(
            cfg, "zone_da", "assimilation", f"sample_{sid:06d}.npz"
        ).exists():
            rows.append(row)
    return rows


def make_model(cfg, geo, method: str):
    s = cfg["superresolution"]
    k = num_zones_from_geo(geo)
    condition = ConditionEncoder(
        history=int(cfg["model"]["history_length"]),
        hidden=int(s["condition_hidden"]),
        device_zone=geo["device_zone"],
        num_zones=k,
        graph_blocks=int(s["condition_graph_blocks"]),
    )
    latent_dim = int(cfg["autoencoder"]["latent_dim_per_zone"])
    if method == "4dsrda":
        return DeterministicSR(condition, latent_dim)
    return DiffusionSR(condition, latent_dim, int(s["diffusion_hidden"]))


def load_cases(cfg, method: str, rows):
    target = target_name(method)
    with np.load(output_path(cfg, "autoencoder", target, "latent_normalization_train_only.npz")) as z:
        latent_mean, latent_std = z["mean"], z["std"]
    cases = {}
    expected_members = int(cfg["enkf"]["ensemble_size"])
    for row in rows:
        sid = int(row["sample_id"])
        data = load_prepared(cfg, sid)
        with np.load(output_path(cfg, "zone_da", "assimilation", f"sample_{sid:06d}.npz")) as z:
            assimilation = {key: z[key] for key in z.files}
        if method in STOCHASTIC_METHODS:
            if "analysis_ensemble" not in assimilation:
                raise KeyError(
                    f"sample {sid:06d} has no analysis_ensemble; rerun 01_zone_da.py --stage assimilate"
                )
            ensemble = assimilation["analysis_ensemble"]
            if ensemble.ndim != 3 or ensemble.shape[1] != expected_members:
                raise ValueError(
                    f"sample {sid:06d} analysis_ensemble shape {ensemble.shape}; "
                    f"expected [frames,{expected_members},zones]"
                )
        with np.load(latent_path(cfg, target, sid)) as z:
            latent = z["latent"]
        cases[sid] = {
            "prepared": data,
            "assimilation": assimilation,
            "latent": ((latent - latent_mean[None, None]) / latent_std[None, None]).astype(np.float32),
        }
    return cases


def examples(cases):
    return [(sid, frame) for sid, case in cases.items() for frame in range(1, len(case["latent"]))]


def deterministic_member_id(sid: int, frame: int, members: int, seed: int) -> int:
    return int((int(seed) + 1009 * int(sid) + 9173 * int(frame)) % int(members))


def condition_arrays(case, frame: int, history: int, norm: LowDimNorm,
                     method: str, member_id: Optional[int] = None):
    data, assimilation = case["prepared"], case["assimilation"]
    pdiag = np.diagonal(assimilation["P_analysis"], axis1=1, axis2=2)
    if method in STOCHASTIC_METHODS:
        if member_id is None:
            raise ValueError("member_id is required for ensemble-conditioned diffusion")
        z_source = assimilation["analysis_ensemble"][:, int(member_id), :]
    else:
        z_source = assimilation["z_analysis"]
    return (
        norm.norm_z(history_window(z_source, frame, history)),
        norm.norm_pdiag(history_window(pdiag, frame, history)),
        norm.norm_frame_u(history_window(data["frame_controls"], frame, history)),
        norm.norm_innovation(history_window(assimilation["innovation"], frame, history)),
    )


def make_batch(batch, cases, history: int, norm: LowDimNorm, device: torch.device,
               method: str, rng: Optional[np.random.Generator] = None,
               deterministic: bool = False, seed: int = 0):
    conditions = []
    for sid, frame in batch:
        member_id = None
        if method in STOCHASTIC_METHODS:
            members = cases[sid]["assimilation"]["analysis_ensemble"].shape[1]
            if deterministic:
                member_id = deterministic_member_id(sid, frame, members, seed)
            else:
                if rng is None:
                    raise ValueError("rng is required for stochastic training-member selection")
                member_id = int(rng.integers(0, members))
        conditions.append(condition_arrays(cases[sid], frame, history, norm, method, member_id))
    tensors = tuple(
        torch.as_tensor(np.stack([x[j] for x in conditions]), dtype=torch.float32, device=device)
        for j in range(4)
    )
    target = torch.as_tensor(
        np.stack([cases[sid]["latent"][frame] for sid, frame in batch]),
        dtype=torch.float32,
        device=device,
    )
    return tensors, target


def append_log(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def validate(cfg, rt, model, method: str, cases, h_zone, adjacency, alpha_bar, norm):
    model.eval()
    items = examples(cases)
    total = count = 0
    maximum = int(cfg["train"]["validation_batches"])
    seed = int(cfg["project"]["seed"])
    generator = torch.Generator(device=rt.device)
    generator.manual_seed(seed + 907)
    for bi, batch in enumerate(epoch_batches(
        items,
        int(cfg["train"]["superresolution_batch_size"]),
        seed,
        rt,
    )):
        if maximum > 0 and bi >= maximum:
            break
        condition, target = make_batch(
            batch, cases, int(cfg["model"]["history_length"]), norm, rt.device,
            method, deterministic=True, seed=seed,
        )
        if method == "4dsrda":
            pred = model(*condition, h_zone, adjacency)
            loss = F.mse_loss(pred, target, reduction="sum")
        else:
            timestep = torch.randint(0, len(alpha_bar), (len(batch),), generator=generator, device=rt.device)
            noise = torch.randn(target.shape, generator=generator, device=rt.device)
            a = alpha_bar[timestep].view(-1, 1, 1)
            noisy = torch.sqrt(a) * target + torch.sqrt(1 - a) * noise
            pred = model(noisy, timestep.float() / max(len(alpha_bar) - 1, 1),
                         *condition, h_zone, adjacency)
            loss = F.mse_loss(pred, noise, reduction="sum")
        total += float(loss)
        count += target.numel()
    return distributed_mean(total, count, rt)


def train(cfg, rt, method: str, resume: bool = False):
    geo = load_prepared_geometry(cfg)
    norm = LowDimNorm.load(cfg)
    h_zone = torch.as_tensor(geo["H_zone"], dtype=torch.float32, device=rt.device)
    adjacency = torch.as_tensor(geo["adjacency"], dtype=torch.float32, device=rt.device)
    model = maybe_ddp(make_model(cfg, geo, method), rt)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["superresolution_learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scaler = make_scaler(rt.device, bool(cfg["train"]["amp"]))
    checkpoint = output_path(cfg, "superresolution", method, "model.pt")
    start_epoch, best = 0, math.inf
    if resume and checkpoint.exists():
        state = load_checkpoint(checkpoint, model, optimizer, scaler, cfg, map_location=rt.device)
        start_epoch, best = int(state["epoch"]) + 1, float(state["best"])
    train_rows = available_rows(cfg, method, "train")
    val_rows = available_rows(cfg, method, "validation") or train_rows
    if not train_rows:
        raise RuntimeError(f"No complete cases for {method}")
    train_cases = load_cases(cfg, method, train_rows)
    val_cases = load_cases(cfg, method, val_rows)
    items = examples(train_cases)
    _, alpha_bar = diffusion_schedule(
        int(cfg["diffusion"]["train_steps"]),
        float(cfg["diffusion"]["beta_start"]),
        float(cfg["diffusion"]["beta_end"]),
        rt.device,
    )
    patience = 0
    max_smoke_batches = int(cfg["smoke"]["batches"]) if cfg["_smoke"] else 0
    base_seed = int(cfg["project"]["seed"])
    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
        model.train()
        total = count = 0
        member_rng = np.random.default_rng(base_seed + 100003 * rt.rank + 7919 * epoch)
        for bi, batch in enumerate(epoch_batches(
            items,
            int(cfg["train"]["superresolution_batch_size"]),
            base_seed + epoch,
            rt,
        )):
            if max_smoke_batches and bi >= max_smoke_batches:
                break
            condition, target = make_batch(
                batch, train_cases, int(cfg["model"]["history_length"]), norm, rt.device,
                method, rng=member_rng, deterministic=False, seed=base_seed,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(rt.device, bool(cfg["train"]["amp"])):
                if method == "4dsrda":
                    pred = model(*condition, h_zone, adjacency)
                    loss = F.mse_loss(pred, target)
                else:
                    timestep = torch.randint(0, len(alpha_bar), (len(batch),), device=rt.device)
                    noise = torch.randn_like(target)
                    a = alpha_bar[timestep].view(-1, 1, 1)
                    noisy = torch.sqrt(a) * target + torch.sqrt(1 - a) * noise
                    pred = model(noisy, timestep.float() / max(len(alpha_bar) - 1, 1),
                                 *condition, h_zone, adjacency)
                    loss = F.mse_loss(pred, noise)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            count += 1
        train_loss = distributed_mean(total, count, rt)
        val_loss = validate(cfg, rt, model, method, val_cases, h_zone, adjacency, alpha_bar, norm)
        stop = False
        if rt.main:
            row = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
            append_log(output_path(cfg, "superresolution", method, "training_log.csv"), row)
            print(method, row)
            if val_loss < best:
                best, patience = val_loss, 0
                save_checkpoint(
                    checkpoint, model, optimizer, scaler, epoch, best, cfg,
                    extra={
                        "method": method,
                        "target": target_name(method),
                        "num_zones": num_zones_from_geo(geo),
                        "analysis_condition": "ensemble_member" if method in STOCHASTIC_METHODS else "ensemble_mean",
                    },
                )
            else:
                patience += 1
            stop = patience >= int(cfg["train"]["patience"])
        stop = broadcast_bool(stop, rt)
        barrier(rt)
        if stop:
            break


def main(args):
    cfg, rt = configure_args(args)
    try:
        train(cfg, rt, args.method, args.resume)
    finally:
        finish_runtime(rt)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    cli_common(parser)
    parser.add_argument("--method", choices=["4dsrda", "diffsrda", "proposed"], required=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
