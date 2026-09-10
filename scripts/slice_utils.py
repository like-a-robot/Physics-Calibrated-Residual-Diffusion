"""Thick-slice selection, obstacle masking, interpolation, and panel drawing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay


def load_obstacle_boxes(cfg: Mapping, slice_y: float, half_thickness: float) -> list[dict]:
    path = Path(cfg["data"]["zone_config_json"])
    if not path.exists():
        raise FileNotFoundError(f"zone_config_json is required for obstacle masking: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    slab_lo = float(slice_y - half_thickness)
    slab_hi = float(slice_y + half_thickness)
    boxes = []
    for box in payload.get("boxes", []):
        if str(box.get("zone_group", "")).lower() != "obstacle":
            continue
        if float(box["y1"]) < slab_lo or float(box["y0"]) > slab_hi:
            continue
        boxes.append(box)
    return boxes


def domain_bounds_from_config(cfg: Mapping, centers: np.ndarray) -> tuple[float, float, float, float]:
    path = Path(cfg["data"]["zone_config_json"])
    if path.exists():
        with path.open("r", encoding="utf-8-sig") as f:
            boxes = json.load(f).get("boxes", [])
        if boxes:
            return (
                min(float(b["x0"]) for b in boxes),
                max(float(b["x1"]) for b in boxes),
                min(float(b["z0"]) for b in boxes),
                max(float(b["z1"]) for b in boxes),
            )
    return (
        float(np.min(centers[:, 0])),
        float(np.max(centers[:, 0])),
        float(np.min(centers[:, 2])),
        float(np.max(centers[:, 2])),
    )


def select_thick_slice(
    centers: np.ndarray,
    slice_y: float,
    half_thickness: float,
    max_cells: int,
) -> np.ndarray:
    """Select cells in |y-slice_y| <= half_thickness with spatially balanced downsampling."""
    distance = np.abs(np.asarray(centers[:, 1], dtype=np.float64) - float(slice_y))
    candidates = np.flatnonzero(distance <= float(half_thickness))
    if len(candidates) < 3:
        raise RuntimeError(
            f"Only {len(candidates)} cells lie in y={slice_y} ± {half_thickness} m. "
            "Increase --slice-half-thickness-m."
        )
    if max_cells <= 0 or len(candidates) <= max_cells:
        return candidates.astype(np.int64)

    x = centers[candidates, 0]
    z = centers[candidates, 2]
    x_span = max(float(np.ptp(x)), 1e-9)
    z_span = max(float(np.ptp(z)), 1e-9)
    nx = max(2, int(np.sqrt(max_cells * x_span / z_span)))
    nz = max(2, int(np.ceil(max_cells / nx)))
    ix = np.clip(((x - x.min()) / x_span * nx).astype(np.int64), 0, nx - 1)
    iz = np.clip(((z - z.min()) / z_span * nz).astype(np.int64), 0, nz - 1)
    keys = ix * nz + iz

    order = np.lexsort((candidates, distance[candidates], keys))
    sorted_candidates = candidates[order]
    sorted_keys = keys[order]
    first = np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
    selected = sorted_candidates[first]
    if len(selected) > max_cells:
        selected = selected[:max_cells]
    return np.sort(selected.astype(np.int64))


def prepare_unique_xz(coords: np.ndarray, decimals: int = 6):
    points = np.round(np.asarray(coords[:, [0, 2]], dtype=np.float64), decimals=decimals)
    unique_points, inverse = np.unique(points, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    return unique_points, inverse, counts


def aggregate_duplicate_values(values: np.ndarray, inverse: np.ndarray, counts: np.ndarray) -> np.ndarray:
    sums = np.bincount(inverse, weights=np.asarray(values, dtype=np.float64), minlength=len(counts))
    return sums / counts


def make_regular_grid(bounds: tuple[float, float, float, float], grid_nx: int, grid_nz: int):
    x0, x1, z0, z1 = bounds
    gx = np.linspace(x0, x1, int(grid_nx), dtype=np.float64)
    gz = np.linspace(z0, z1, int(grid_nz), dtype=np.float64)
    xx, zz = np.meshgrid(gx, gz)
    return gx, gz, xx, zz


def obstacle_mask(xx: np.ndarray, zz: np.ndarray, obstacles: Sequence[Mapping]) -> np.ndarray:
    mask = np.zeros(xx.shape, dtype=bool)
    for box in obstacles:
        mask |= (
            (xx >= float(box["x0"]))
            & (xx <= float(box["x1"]))
            & (zz >= float(box["z0"]))
            & (zz <= float(box["z1"]))
        )
    return mask


def interpolate_fields(
    coords: np.ndarray,
    fields: Mapping[str, np.ndarray],
    xx: np.ndarray,
    zz: np.ndarray,
    obstacles: Sequence[Mapping],
) -> Dict[str, np.ndarray]:
    """Interpolate irregular CFD cells to one complete x-z raster."""
    unique_points, inverse, counts = prepare_unique_xz(coords)
    if len(unique_points) < 3:
        raise RuntimeError("Fewer than three unique x-z points are available for interpolation")
    triangulation = Delaunay(unique_points)
    mask = obstacle_mask(xx, zz, obstacles)
    output: Dict[str, np.ndarray] = {}
    for name, values in fields.items():
        unique_values = aggregate_duplicate_values(values, inverse, counts)
        linear = np.asarray(LinearNDInterpolator(triangulation, unique_values)(xx, zz), dtype=np.float64)
        nearest = np.asarray(NearestNDInterpolator(unique_points, unique_values)(xx, zz), dtype=np.float64)
        raster = np.where(np.isfinite(linear), linear, nearest)
        raster[mask] = np.nan
        output[name] = raster.astype(np.float32)
    return output


def add_obstacle_patches(ax, obstacles: Iterable[Mapping]) -> None:
    for box in obstacles:
        ax.add_patch(
            Rectangle(
                (float(box["x0"]), float(box["z0"])),
                float(box["x1"]) - float(box["x0"]),
                float(box["z1"]) - float(box["z0"]),
                facecolor="white",
                edgecolor="black",
                linewidth=0.65,
                zorder=6,
            )
        )


def draw_panel(
    ax,
    raster: np.ndarray,
    bounds: tuple[float, float, float, float],
    sensor_xyz: np.ndarray,
    obstacles: Sequence[Mapping],
    vmin: float,
    vmax: float,
    cmap_name: str,
    xlabel: str,
    ylabel: str,
):
    x0, x1, z0, z1 = bounds
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")
    image = ax.imshow(
        np.ma.masked_invalid(raster),
        origin="lower",
        extent=(x0, x1, z0, z1),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        aspect="equal",
    )
    add_obstacle_patches(ax, obstacles)
    ax.scatter(
        sensor_xyz[:, 0],
        sensor_xyz[:, 2],
        s=18,
        c="red",
        marker="o",
        edgecolors="white",
        linewidths=0.5,
        zorder=8,
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(z0, z1)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_ylabel(ylabel)
    ax.tick_params(direction="out", length=2.7)
    return image
