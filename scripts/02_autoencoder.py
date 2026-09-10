"""Train full-zone full-field and residual autoencoders without anchor sampling."""
from __future__ import annotations

import argparse
import csv
import math

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    ZonedFieldAutoencoder, autocast_context, barrier, broadcast_bool, cli_common,
    configure_args, distributed_mean, epoch_batches, finish_runtime, independent_rows,
    iter_chunks, load_checkpoint, load_prepared, load_prepared_geometry, make_scaler,
    maybe_ddp, num_zones_from_geo, output_path, read_manifest, save_checkpoint,
    temperature_cache_path, unwrap,
)


def rows_with_cache(cfg, split=None):
    out = []
    for row in read_manifest(cfg):
        if split is not None and row["split"] != split:
            continue
        sid = int(row["sample_id"])
        if temperature_cache_path(cfg, sid).exists():
            out.append(row)
    return out


def make_model(cfg, geo):
    a = cfg["autoencoder"]
    return ZonedFieldAutoencoder(
        num_zones=num_zones_from_geo(geo), latent_dim=int(a["latent_dim_per_zone"]),
        point_hidden=int(a["point_hidden"]), decoder_hidden=int(a["decoder_hidden"]),
        frequencies=int(a["coord_fourier_frequencies"]),
        zone_embedding_dim=int(a["zone_embedding_dim"]),
    )


def field_stats_path(cfg, target):
    return output_path(cfg, "autoencoder", target, "field_stats_train_only.npz")


def latent_path(cfg, target, sid):
    return output_path(cfg, "autoencoder", target, "latents", f"sample_{sid:06d}.npz")


def zone_indices(geo, zid0):
    order, offsets = geo["zone_order"], geo["zone_offsets"]
    return order[int(offsets[zid0]):int(offsets[zid0 + 1])]


def raw_zone_values(target, temperature, data, frame, zid0, idx):
    values = np.asarray(temperature[frame, idx], dtype=np.float32)
    if target == "residual":
        values = values - float(data["z_true"][frame, zid0])
    return values


def fit_field_stats(cfg, target):
    path = field_stats_path(cfg, target)
    if path.exists():
        return
    geo = load_prepared_geometry(cfg); total = squared = 0.0; count = 0
    chunk = int(cfg["data"]["cell_chunk"])
    for row in rows_with_cache(cfg, "train"):
        sid = int(row["sample_id"]); data = load_prepared(cfg, sid)
        temp = np.load(temperature_cache_path(cfg, sid), mmap_mode="r")
        for frame in range(len(data["times_s"])):
            for sl in iter_chunks(temp.shape[1], chunk):
                idx = np.arange(sl.start, sl.stop, dtype=np.int64)
                values = np.asarray(temp[frame, sl], dtype=np.float64)
                if target == "residual":
                    values -= data["z_true"][frame, geo["zone_id"][sl] - 1]
                total += float(values.sum()); squared += float(np.dot(values, values)); count += len(values)
    mean = total / count; std = math.sqrt(max(squared / count - mean * mean, 1e-12))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=np.float32(mean), std=np.float32(std), count=np.int64(count))
    print(f"{target} field stats: mean={mean:.6g}, std={std:.6g}")


def load_field_stats(cfg, target):
    with np.load(field_stats_path(cfg, target)) as z:
        return float(z["mean"]), float(z["std"])


def examples(cfg, rows, geo):
    k = num_zones_from_geo(geo); items = []
    for row in rows:
        sid = int(row["sample_id"]); data = load_prepared(cfg, sid)
        items.extend((sid, frame, zid0) for frame in range(len(data["times_s"])) for zid0 in range(k))
    return items


def load_case(cfg, sid, cache):
    if sid not in cache:
        cache[sid] = (load_prepared(cfg, sid), np.load(temperature_cache_path(cfg, sid), mmap_mode="r"))
    return cache[sid]


def tensors_for_example(cfg, geo, target, item, mean, std, cache, device):
    sid, frame, zid0 = item; data, temp = load_case(cfg, sid, cache); idx = zone_indices(geo, zid0)
    values = (raw_zone_values(target, temp, data, frame, zid0, idx) - mean) / std
    return (
        torch.as_tensor(values, dtype=torch.float32, device=device),
        torch.as_tensor(geo["local_coords"][idx], dtype=torch.float32, device=device),
        torch.as_tensor(geo["volumes"][idx], dtype=torch.float32, device=device),
    )


def append_log(path, row):
    path.parent.mkdir(parents=True, exist_ok=True); exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists: w.writeheader()
        w.writerow(row)


@torch.no_grad()
def validate(cfg, rt, model, target, items, geo, mean, std):
    model.eval(); cache = {}; total = count = 0
    maximum = min(int(cfg["train"]["validation_examples"]), len(items))
    rng = np.random.default_rng(int(cfg["project"]["seed"]) + (17 if target == "full" else 19))
    chosen = [items[int(i)] for i in rng.choice(len(items), maximum, replace=False)] if maximum < len(items) else items
    for item in chosen[rt.rank::rt.world_size]:
        values, coords, volumes = tensors_for_example(cfg, geo, target, item, mean, std, cache, rt.device)
        prediction, _ = unwrap(model)(values, coords, volumes, item[2],
                                      int(cfg["autoencoder"]["encoder_cell_chunk"]),
                                      int(cfg["autoencoder"]["decoder_cell_chunk"]))
        total += float(F.mse_loss(prediction, values, reduction="sum")); count += len(values)
    return distributed_mean(total, count, rt)


def train(cfg, rt, target, resume=False):
    geo = load_prepared_geometry(cfg); mean, std = load_field_stats(cfg, target)
    model = maybe_ddp(make_model(cfg, geo), rt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["autoencoder_learning_rate"]),
                                  weight_decay=float(cfg["train"]["weight_decay"]))
    scaler = make_scaler(rt.device, bool(cfg["train"]["amp"]))
    checkpoint = output_path(cfg, "autoencoder", target, "model.pt")
    start_epoch, best = 0, math.inf
    if resume and checkpoint.exists():
        state = load_checkpoint(checkpoint, model, optimizer, scaler, cfg, map_location=rt.device)
        start_epoch, best = int(state["epoch"]) + 1, float(state["best"])
    train_items = examples(cfg, rows_with_cache(cfg, "train"), geo)
    val_items = examples(cfg, rows_with_cache(cfg, "validation") or rows_with_cache(cfg, "train"), geo)
    if not train_items: raise RuntimeError("No AE training examples")
    steps = min(int(cfg["train"]["autoencoder_steps_per_epoch"]), len(train_items))
    patience = 0; cache = {}
    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
        model.train(); rng = np.random.default_rng(int(cfg["project"]["seed"]) + 1000 * epoch + (0 if target == "full" else 1))
        chosen = [train_items[int(i)] for i in rng.choice(len(train_items), steps, replace=False)]
        total = count = 0
        for batch in epoch_batches(chosen, 1, int(cfg["project"]["seed"]) + epoch, rt):
            item = batch[0]; values, coords, volumes = tensors_for_example(cfg, geo, target, item, mean, std, cache, rt.device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(rt.device, bool(cfg["train"]["amp"])):
                prediction, latent = model(values, coords, volumes, item[2],
                                           int(cfg["autoencoder"]["encoder_cell_chunk"]),
                                           int(cfg["autoencoder"]["decoder_cell_chunk"]))
                rec = F.mse_loss(prediction, values)
                loss = rec + float(cfg["autoencoder"]["latent_l2_weight"]) * latent.pow(2).mean()
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            scaler.step(optimizer); scaler.update()
            total += float(rec.detach()); count += 1
        train_loss = distributed_mean(total, count, rt)
        val_loss = validate(cfg, rt, model, target, val_items, geo, mean, std)
        stop = False
        if rt.main:
            row = {"epoch": epoch + 1, "train_standardized_mse": train_loss,
                   "val_standardized_mse": val_loss, "val_rmse_C": math.sqrt(val_loss) * std}
            append_log(output_path(cfg, "autoencoder", target, "training_log.csv"), row); print(target, row)
            if val_loss < best:
                best, patience = val_loss, 0
                save_checkpoint(checkpoint, model, optimizer, scaler, epoch, best, cfg,
                                extra={"target": target, "num_zones": num_zones_from_geo(geo)})
            else: patience += 1
            stop = patience >= int(cfg["train"]["patience"])
        stop = broadcast_bool(stop, rt); barrier(rt)
        if stop: break


@torch.no_grad()
def encode_all(cfg, rt, target):
    geo = load_prepared_geometry(cfg); mean, std = load_field_stats(cfg, target); k = num_zones_from_geo(geo)
    model = make_model(cfg, geo).to(rt.device)
    load_checkpoint(output_path(cfg, "autoencoder", target, "model.pt"), model, cfg=cfg, map_location=rt.device); model.eval()
    for row in independent_rows(rows_with_cache(cfg), rt):
        sid = int(row["sample_id"]); data = load_prepared(cfg, sid); temp = np.load(temperature_cache_path(cfg, sid), mmap_mode="r")
        latent = np.empty((len(data["times_s"]), k, int(cfg["autoencoder"]["latent_dim_per_zone"])), dtype=np.float32)
        for frame in range(len(data["times_s"])):
            for zid0 in range(k):
                idx = zone_indices(geo, zid0)
                values = (raw_zone_values(target, temp, data, frame, zid0, idx) - mean) / std
                h = model.encode_zone(
                    torch.as_tensor(values, dtype=torch.float32, device=rt.device),
                    torch.as_tensor(geo["local_coords"][idx], dtype=torch.float32, device=rt.device),
                    torch.as_tensor(geo["volumes"][idx], dtype=torch.float32, device=rt.device),
                    zid0, int(cfg["autoencoder"]["encoder_cell_chunk"]),
                )
                latent[frame, zid0] = h.cpu().numpy()
        path = latent_path(cfg, target, sid); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, latent=latent, times_s=data["times_s"])
        print(f"encoded {target} sample {sid:06d}: {latent.shape}")
    barrier(rt)


def fit_latent_norm(cfg, rt, target):
    if not rt.main: return
    arrays = []
    for row in rows_with_cache(cfg, "train"):
        with np.load(latent_path(cfg, target, int(row["sample_id"]))) as z: arrays.append(z["latent"])
    x = np.concatenate(arrays, axis=0)
    mean = x.mean(axis=(0, 1)).astype(np.float32); std = np.maximum(x.std(axis=(0, 1)), 1e-6).astype(np.float32)
    np.savez(output_path(cfg, "autoencoder", target, "latent_normalization_train_only.npz"), mean=mean, std=std)


def main(args):
    cfg, rt = configure_args(args)
    try:
        stages = ["fit_stats", "train", "encode", "fit_latent_norm"] if args.stage == "all" else [args.stage]
        for stage in stages:
            if stage == "fit_stats" and rt.main: fit_field_stats(cfg, args.target)
            barrier(rt)
            if stage == "train": train(cfg, rt, args.target, args.resume)
            elif stage == "encode": encode_all(cfg, rt, args.target)
            elif stage == "fit_latent_norm": fit_latent_norm(cfg, rt, args.target)
            barrier(rt)
    finally:
        finish_runtime(rt)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__); cli_common(p)
    p.add_argument("--target", choices=["full", "residual"], required=True)
    p.add_argument("--stage", choices=["fit_stats", "train", "encode", "fit_latent_norm", "all"], default="all")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
