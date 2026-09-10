"""Create compact publication layouts for deterministic CFD slice comparisons.

Outputs
-------
1) Truth evolution: 1 x N_times
   paper_truth_evolution.png / .pdf
2) Method comparison: 4 methods x N_times
   paper_reconstruction_comparison.png / .pdf

Inference is shared with evaluation and benchmarking through inference.py.
Slice geometry, interpolation, and panel drawing are provided by slice_utils.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np

import slice_utils as slices
from inference import DISPLAY, find_manifest_row, load_model_bundle, reconstruct_selected_cells, resolve_sample_id

from common import (
    LowDimNorm,
    barrier,
    cli_common,
    configure_args,
    finish_runtime,
    load_prepared,
    load_prepared_geometry,
    output_path,
    temperature_cache_path,
)


METHODS = ("proposed", "4dsrda", "diffsrda", "zone_const")


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "savefig.dpi": 500,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finite_minmax(all_rasters: Mapping[int, Mapping[str, np.ndarray]]) -> tuple[float, float]:
    values = []
    for rasters in all_rasters.values():
        for name in ("truth", *METHODS):
            arr = np.asarray(rasters[name])
            arr = arr[np.isfinite(arr)]
            if arr.size:
                values.append(arr)
    if not values:
        raise ValueError("All raster values are NaN")
    merged = np.concatenate(values)
    vmin = float(np.min(merged))
    vmax = float(np.max(merged))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def save_both(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)


def draw_compact_panel(
    ax,
    raster: np.ndarray,
    bounds,
    sensor_xyz: np.ndarray,
    obstacles,
    vmin: float,
    vmax: float,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
):
    image = slices.draw_panel(
        ax,
        raster,
        bounds,
        sensor_xyz,
        obstacles,
        vmin,
        vmax,
        "turbo",
        "x (m)" if show_xlabel else "",
        "z (m)" if show_ylabel else "",
    )
    if not show_xlabel:
        ax.tick_params(labelbottom=False)
    if not show_ylabel:
        ax.tick_params(labelleft=False)
    return image


def plot_truth_evolution(
    output_dir: Path,
    times: list[int],
    all_rasters: Mapping[int, Mapping[str, np.ndarray]],
    bounds,
    sensor_xyz: np.ndarray,
    obstacles,
    vmin: float,
    vmax: float,
):
    n = len(times)
    # Wide panels: each physical slice is about 25 m x 12 m.
    fig = plt.figure(figsize=(3.65 * n + 0.55, 2.35))
    gs = fig.add_gridspec(
        1,
        n + 1,
        width_ratios=[1.0] * n + [0.035],
        left=0.065,
        right=0.975,
        bottom=0.20,
        top=0.89,
        wspace=0.10,
    )

    image = None
    letters = "abcdefghijklmnopqrstuvwxyz"
    for j, time_s in enumerate(times):
        ax = fig.add_subplot(gs[0, j])
        image = draw_compact_panel(
            ax,
            all_rasters[time_s]["truth"],
            bounds,
            sensor_xyz,
            obstacles,
            vmin,
            vmax,
            show_xlabel=True,
            show_ylabel=(j == 0),
        )
        ax.set_title(f"({letters[j]})  t = {time_s} s", fontweight="bold", pad=4)

    cax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Temperature (°C)", labelpad=5)
    cbar.ax.tick_params(labelsize=7.5)

    save_both(fig, output_dir / "paper_truth_evolution")
    plt.close(fig)


def plot_method_comparison(
    output_dir: Path,
    times: list[int],
    all_rasters: Mapping[int, Mapping[str, np.ndarray]],
    bounds,
    sensor_xyz: np.ndarray,
    obstacles,
    vmin: float,
    vmax: float,
):
    # Layout requested for the paper: 4 rows (methods) × 3 columns (times)
    nrows = len(METHODS)
    ncols = len(times)

    fig = plt.figure(figsize=(9.6, 1.55 * nrows + 0.55))
    gs = fig.add_gridspec(
        nrows,
        ncols + 1,
        width_ratios=[1.0] * ncols + [0.035],
        left=0.10,
        right=0.975,
        bottom=0.105,
        top=0.915,
        wspace=0.075,
        hspace=0.16,
    )

    image = None
    for row, method in enumerate(METHODS):
        for col, time_s in enumerate(times):
            ax = fig.add_subplot(gs[row, col])
            image = draw_compact_panel(
                ax,
                all_rasters[time_s][method],
                bounds,
                sensor_xyz,
                obstacles,
                vmin,
                vmax,
                show_xlabel=(row == nrows - 1),
                show_ylabel=(col == 0),
            )

            # Time labels only once, at the top of each column.
            if row == 0:
                ax.set_title(f"t = {time_s} s", fontweight="bold", pad=4)

            # Method labels only once, along the left side of each row.
            if col == 0:
                ax.text(
                    -0.23,
                    0.50,
                    DISPLAY[method],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=9.0,
                    fontweight="bold",
                )

    # One colorbar shared by all 12 panels.
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Temperature (°C)", labelpad=5)
    cbar.ax.tick_params(labelsize=7.5)

    save_both(fig, output_dir / "paper_reconstruction_comparison")
    plt.close(fig)


def main(args) -> None:
    cfg, rt = configure_args(args)
    try:
        if not rt.main:
            barrier(rt)
            return

        configure_plotting()

        sid = resolve_sample_id(cfg, args.sample_id)
        find_manifest_row(cfg, sid)
        geo = load_prepared_geometry(cfg)
        data = load_prepared(cfg, sid)
        norm = LowDimNorm.load(cfg)

        centers = np.asarray(geo["centers"], dtype=np.float64)
        sensor_xyz = np.asarray(geo["sensor_xyz"], dtype=np.float64)
        selected = slices.select_thick_slice(
            centers,
            float(args.slice_height_m),
            float(args.slice_half_thickness_m),
            int(args.max_slice_cells),
        )
        obstacles = slices.load_obstacle_boxes(
            cfg,
            float(args.slice_height_m),
            float(args.slice_half_thickness_m),
        )
        bounds = slices.domain_bounds_from_config(cfg, centers)
        _, _, xx, zz = slices.make_regular_grid(
            bounds,
            int(args.grid_nx),
            int(args.grid_nz),
        )

        with np.load(output_path(cfg, "zone_da", "assimilation", f"sample_{sid:06d}.npz")) as z:
            assimilation = {name: z[name] for name in z.files}
        truth_all = np.load(temperature_cache_path(cfg, sid), mmap_mode="r")
        bundles = {
            method: load_model_bundle(cfg, geo, method, rt.device)
            for method in METHODS
        }

        time_to_frame = {int(t): i for i, t in enumerate(data["times_s"])}
        requested_times = [int(t) for t in (
            args.times if args.times is not None else cfg["reconstruct"]["save_frames"]
        )]
        if args.smoke and args.times is None:
            requested_times = [t for t in requested_times if t in time_to_frame]
            if not requested_times:
                requested_times = [int(data["times_s"][-1])]
        if not requested_times:
            raise ValueError("At least one plot time is required")
        for t in requested_times:
            if t not in time_to_frame:
                raise ValueError(f"t={t} s is not present in sample {sid:06d}")

        all_rasters: Dict[int, Dict[str, np.ndarray]] = {}
        print(
            f"sample={sid:06d}; selected {len(selected)} CFD cells in "
            f"y={args.slice_height_m} ± {args.slice_half_thickness_m} m"
        )

        for time_s in requested_times:
            frame = time_to_frame[time_s]
            fields: Dict[str, np.ndarray] = {
                "truth": np.asarray(truth_all[frame, selected], dtype=np.float32)
            }
            for method in METHODS:
                mean, _ = reconstruct_selected_cells(
                    cfg,
                    geo,
                    data,
                    assimilation,
                    bundles[method],
                    frame,
                    selected,
                    norm,
                    rt.device,
                )
                fields[method] = mean
                print(f"reconstructed {method}, t={time_s} s")

            all_rasters[time_s] = slices.interpolate_fields(
                centers[selected], fields, xx, zz, obstacles
            )

        # IMPORTANT: one global color range over all truth/method/time panels.
        if args.vmin is None or args.vmax is None:
            auto_vmin, auto_vmax = finite_minmax(all_rasters)
            vmin = auto_vmin if args.vmin is None else float(args.vmin)
            vmax = auto_vmax if args.vmax is None else float(args.vmax)
        else:
            vmin, vmax = float(args.vmin), float(args.vmax)
        if vmax <= vmin:
            raise ValueError(f"Invalid color range: vmin={vmin}, vmax={vmax}")

        output_dir = output_path(
            cfg,
            "evaluation",
            "paper_slice_layout",
            f"sample_{sid:06d}",
            mkdir=True,
        )

        plot_truth_evolution(
            output_dir,
            requested_times,
            all_rasters,
            bounds,
            sensor_xyz,
            obstacles,
            vmin,
            vmax,
        )
        plot_method_comparison(
            output_dir,
            requested_times,
            all_rasters,
            bounds,
            sensor_xyz,
            obstacles,
            vmin,
            vmax,
        )

        print(f"saved publication layouts to: {output_dir}")
        print(f"shared temperature range: [{vmin:.3f}, {vmax:.3f}] °C")
    finally:
        finish_runtime(rt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    cli_common(parser)
    parser.add_argument(
        "--sample-id",
        type=int,
        default=None,
        help="Defaults to evaluation/best_proposed_case.json.",
    )
    parser.add_argument(
        "--times",
        type=int,
        nargs="+",
        default=None,
        help="Column times; defaults to reconstruct.save_frames in config.yaml.",
    )
    parser.add_argument("--slice-height-m", type=float, default=3.1)
    parser.add_argument("--slice-half-thickness-m", type=float, default=0.15)
    parser.add_argument(
        "--max-slice-cells",
        type=int,
        default=100000,
        help="0 keeps every slab cell; positive values spatially downsample.",
    )
    parser.add_argument("--grid-nx", type=int, default=700)
    parser.add_argument("--grid-nz", type=int, default=336)
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Optional fixed lower temperature limit. Default: global minimum.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Optional fixed upper temperature limit. Default: global maximum.",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
