"""Merge reconstruction shards, evaluate methods, and create publication-quality figures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from common import cli_common, configure_args, finish_runtime, output_path

METHODS = ("zone_const", "4dsrda", "diffsrda", "proposed")
STOCHASTIC_METHODS = ("diffsrda", "proposed")
DISPLAY = {
    "zone_const": "Zone-Analysis-Const",
    "4dsrda": "4D-SRDA",
    "diffsrda": "DiffSRDA",
    "proposed": "Proposed",
}
COLORS = {
    "zone_const": "#7A7A7A",
    "4dsrda": "#3B6FB6",
    "diffsrda": "#E08B2C",
    "proposed": "#B33A3A",
}
LINESTYLES = {
    "zone_const": "--",
    "4dsrda": "-.",
    "diffsrda": ":",
    "proposed": "-",
}


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_method(cfg, method: str):
    shard_dir = output_path(cfg, "reconstruction", method, "shards")
    paths = sorted(shard_dir.glob("rank_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No shards for {method}: {shard_dir}")

    values = {
        "sample_ids": [],
        "full_rmse": [],
        "hotspot_rmse": [],
        "coverage_volume": [],
        "total_volume": [],
        "rank_volume": [],
    }
    times = levels = None
    for path in paths:
        with np.load(path) as z:
            for name in values:
                values[name].append(z[name])
            if times is None:
                times = z["times_s"]
            if levels is None:
                levels = z["nominal_levels"]

    merged = {
        name: np.concatenate(parts, axis=0) if parts else np.empty(0)
        for name, parts in values.items()
    }
    order = np.argsort(merged["sample_ids"])
    for name in values:
        merged[name] = merged[name][order]
    return merged, times, levels


def method_statistics(data, times):
    valid = slice(1, None)
    full = data["full_rmse"][:, valid]
    hot = data["hotspot_rmse"][:, valid]
    trajectory_full = np.sqrt(np.mean(full ** 2, axis=1))
    trajectory_hot = np.sqrt(np.mean(hot ** 2, axis=1))
    return {
        "trajectory_full": trajectory_full,
        "trajectory_hot": trajectory_hot,
        "full_mean": float(trajectory_full.mean()),
        "full_std": float(trajectory_full.std(ddof=1) if len(trajectory_full) > 1 else 0.0),
        "hot_mean": float(trajectory_hot.mean()),
        "hot_std": float(trajectory_hot.std(ddof=1) if len(trajectory_hot) > 1 else 0.0),
        "times": times[valid],
        "full_time_mean": full.mean(axis=0),
        "full_time_std": full.std(axis=0, ddof=1) if len(full) > 1 else np.zeros(full.shape[1]),
        "hot_time_mean": hot.mean(axis=0),
        "hot_time_std": hot.std(axis=0, ddof=1) if len(hot) > 1 else np.zeros(hot.shape[1]),
    }


def uncertainty_statistics(data, levels):
    total_volume = float(np.sum(data["total_volume"]))
    if total_volume <= 0.0:
        return None
    coverage = np.sum(data["coverage_volume"], axis=0) / total_volume
    rank_volume = np.sum(data["rank_volume"], axis=0)
    rank_probability = rank_volume / max(float(rank_volume.sum()), 1e-30)
    return {
        "levels": np.asarray(levels, dtype=np.float64),
        "coverage": coverage.astype(np.float64),
        "rank_probability": rank_probability.astype(np.float64),
    }


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.8)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.65)


def detailed_ylim(curves, field: str) -> tuple[float, float]:
    lows, highs = [], []
    for method in ("4dsrda", "diffsrda", "proposed"):
        stats = curves[method]
        mean = stats[f"{field}_time_mean"]
        std = stats[f"{field}_time_std"]
        lows.append(float(np.nanmin(mean - std)))
        highs.append(float(np.nanmax(mean + std)))
    lo = max(0.0, min(lows) - 0.05 * (max(highs) - min(lows)))
    hi = max(highs) + 0.08 * max(max(highs) - lo, 1e-6)
    return lo, hi


def plot_deterministic_curves(curves, eval_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.15))
    specs = (
        ("full", "Full-field RMSE (°C)", "(a) Full-field reconstruction"),
        ("hot", "Top 5% hotspot RMSE (°C)", "(b) Hotspot reconstruction"),
    )

    legend_handles = []
    for ax, (field, ylabel, title) in zip(axes, specs):
        for method in ("4dsrda", "diffsrda", "proposed"):
            stats = curves[method]
            time_s = stats["times"]
            mean = stats[f"{field}_time_mean"]
            std = stats[f"{field}_time_std"]
            line, = ax.plot(
                time_s,
                mean,
                label=DISPLAY[method],
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=2.1 if method == "proposed" else 1.65,
                zorder=3 if method == "proposed" else 2,
            )
            ax.fill_between(
                time_s,
                np.maximum(mean - std, 0.0),
                mean + std,
                color=COLORS[method],
                alpha=0.10 if method == "proposed" else 0.075,
                linewidth=0,
                zorder=1,
            )
            if len(legend_handles) < 3:
                legend_handles.append(line)

        ax.set_xlim(float(curves["proposed"]["times"][0]), float(curves["proposed"]["times"][-1]))
        ax.set_ylim(*detailed_ylim(curves, field))
        ax.set_xticks(np.arange(0, 1801, 300))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold", pad=6)
        clean_axis(ax)

        inset = inset_axes(ax, width="34%", height="41%", loc="upper right", borderpad=0.85)
        zone = curves["zone_const"]
        inset.plot(
            zone["times"],
            zone[f"{field}_time_mean"],
            color=COLORS["zone_const"],
            linestyle=LINESTYLES["zone_const"],
            linewidth=1.25,
        )
        inset.fill_between(
            zone["times"],
            np.maximum(zone[f"{field}_time_mean"] - zone[f"{field}_time_std"], 0.0),
            zone[f"{field}_time_mean"] + zone[f"{field}_time_std"],
            color=COLORS["zone_const"],
            alpha=0.10,
            linewidth=0,
        )
        inset.set_xlim(float(zone["times"][0]), float(zone["times"][-1]))
        inset.set_ylim(0.0, float(np.max(zone[f"{field}_time_mean"] + zone[f"{field}_time_std"])) * 1.08)
        inset.set_xticks([0, 900, 1800])
        inset.tick_params(labelsize=6.6, direction="out", length=2.0)
        inset.set_title("Zone-Analysis-Const", fontsize=6.8, pad=2)
        inset.spines["top"].set_visible(False)
        inset.spines["right"].set_visible(False)

    fig.legend(
        legend_handles,
        [DISPLAY[m] for m in ("4dsrda", "diffsrda", "proposed")],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.5,
    )
    fig.text(0.5, 0.012, "Shaded regions denote ±1 standard deviation across test trajectories.",
             ha="center", va="bottom", fontsize=7.4)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.82, wspace=0.30)
    fig.savefig(eval_dir / "deterministic_rmse_vs_time.png", bbox_inches="tight")
    fig.savefig(eval_dir / "deterministic_rmse_vs_time.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_coverage_reliability(uncertainty, eval_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.25, 3.65))
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.2, label="Ideal calibration")
    for method in STOCHASTIC_METHODS:
        stats = uncertainty[method]
        ax.plot(
            stats["levels"],
            stats["coverage"],
            marker="o" if method == "diffsrda" else "s",
            markersize=5.2,
            linewidth=1.8 if method == "proposed" else 1.55,
            color=COLORS[method],
            label=DISPLAY[method],
        )
    ax.set_xlim(0.45, 0.93)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.5, 0.8, 0.9])
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Volume-weighted empirical coverage")
    clean_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(eval_dir / "coverage_reliability.png", bbox_inches="tight")
    fig.savefig(eval_dir / "coverage_reliability.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_rank_histograms(uncertainty, eval_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, method, panel in zip(axes, STOCHASTIC_METHODS, ("(a)", "(b)")):
        probability = uncertainty[method]["rank_probability"]
        ranks = np.arange(len(probability))
        ax.bar(ranks, probability, width=0.86, color=COLORS[method], alpha=0.82, linewidth=0)
        ax.axhline(1.0 / len(probability), color="#4F4F4F", linestyle="--", linewidth=1.1)
        ax.set_xlim(-0.8, len(probability) - 0.2)
        ax.set_xticks([0, (len(probability) - 1) // 2, len(probability) - 1])
        ax.set_xlabel("Rank of CFD reference")
        ax.set_title(f"{panel} {DISPLAY[method]}", loc="left", fontweight="bold", pad=5)
        clean_axis(ax)
    axes[0].set_ylabel("Volume-weighted relative frequency")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.19, top=0.87, wspace=0.20)
    fig.savefig(eval_dir / "rank_histograms.png", bbox_inches="tight")
    fig.savefig(eval_dir / "rank_histograms.pdf", bbox_inches="tight")
    plt.close(fig)


def summarize(cfg):
    configure_plotting()
    eval_dir = output_path(cfg, "evaluation", mkdir=True)
    main_rows, trajectory_rows, time_rows = [], [], []
    coverage_rows, rank_rows = [], []
    loaded, curves, uncertainty = {}, {}, {}

    for method in METHODS:
        data, times, levels = load_method(cfg, method)
        stats = method_statistics(data, times)
        loaded[method] = (data, times, levels, stats)
        curves[method] = stats

        main_rows.append({
            "method": DISPLAY[method],
            "full_field_RMSE_mean_C": stats["full_mean"],
            "full_field_RMSE_std_C": stats["full_std"],
            "top5_hotspot_RMSE_mean_C": stats["hot_mean"],
            "top5_hotspot_RMSE_std_C": stats["hot_std"],
            "test_trajectories": len(stats["trajectory_full"]),
        })
        for i, sid in enumerate(data["sample_ids"]):
            trajectory_rows.append({
                "method": DISPLAY[method],
                "sample_id": int(sid),
                "full_field_RMSE_C": float(stats["trajectory_full"][i]),
                "top5_hotspot_RMSE_C": float(stats["trajectory_hot"][i]),
            })
        for j, time_s in enumerate(stats["times"]):
            time_rows.append({
                "method": DISPLAY[method],
                "time_s": int(time_s),
                "full_field_RMSE_mean_C": float(stats["full_time_mean"][j]),
                "full_field_RMSE_std_C": float(stats["full_time_std"][j]),
                "top5_hotspot_RMSE_mean_C": float(stats["hot_time_mean"][j]),
                "top5_hotspot_RMSE_std_C": float(stats["hot_time_std"][j]),
            })

        if method in STOCHASTIC_METHODS:
            uq = uncertainty_statistics(data, levels)
            if uq is None:
                raise RuntimeError(f"No uncertainty statistics were accumulated for {method}")
            uncertainty[method] = uq
            for nominal, empirical in zip(uq["levels"], uq["coverage"]):
                coverage_rows.append({
                    "method": DISPLAY[method],
                    "nominal_coverage": float(nominal),
                    "volume_weighted_empirical_coverage": float(empirical),
                })
            for rank, probability in enumerate(uq["rank_probability"]):
                rank_rows.append({
                    "method": DISPLAY[method],
                    "rank": int(rank),
                    "volume_weighted_relative_frequency": float(probability),
                })

    write_csv(eval_dir / "main_results.csv", main_rows)
    write_csv(eval_dir / "per_trajectory_metrics.csv", trajectory_rows)
    write_csv(eval_dir / "metrics_vs_time.csv", time_rows)
    write_csv(eval_dir / "uncertainty_coverage.csv", coverage_rows)
    write_csv(eval_dir / "rank_histogram.csv", rank_rows)

    strict_data, _, _, strict_stats = loaded["proposed"]
    best_index = int(np.argmin(strict_stats["trajectory_full"]))
    best_sid = int(strict_data["sample_ids"][best_index])
    best_record = {
        "sample_id": best_sid,
        "trajectory_full_field_RMSE_C": float(strict_stats["trajectory_full"][best_index]),
        "selection_method": "Proposed ensemble mean",
        "selection_rule": "minimum trajectory-level full-field RMSE over the test set, evaluated at 30-1800 s",
    }
    with (eval_dir / "best_proposed_case.json").open("w", encoding="utf-8") as f:
        json.dump(best_record, f, ensure_ascii=False, indent=2)

    summary = {
        "deterministic_results": main_rows,
        "uncertainty_coverage": coverage_rows,
        "best_case": best_record,
    }
    with (eval_dir / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_deterministic_curves(curves, eval_dir)
    plot_coverage_reliability(uncertainty, eval_dir)
    plot_rank_histograms(uncertainty, eval_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(args):
    cfg, rt = configure_args(args)
    try:
        if rt.main:
            summarize(cfg)
    finally:
        finish_runtime(rt)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    cli_common(parser)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
