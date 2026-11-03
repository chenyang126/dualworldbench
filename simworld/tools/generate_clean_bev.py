#!/usr/bin/env python3
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image, ImageFilter


REPO_ROOT = Path(__file__).resolve().parents[2]
MESH_ROOT = REPO_ROOT / "inputs/mesh"
OUTPUT_ROOT = REPO_ROOT / "outputs/generated/bev"


@dataclass(frozen=True)
class SceneConfig:
    resolution: float = 0.1
    margin_m: float = 2.0
    ground_hint_z: float = 0.0
    ground_search_half_window: float = 1.5
    min_obstacle_height_m: float = 0.08
    max_obstacle_height_m: float = 2.5
    min_hits_per_cell: int = 1
    thicken_kernel_px: int = 3
    close_kernel_px: int = 5
    open_kernel_px: int = 1


SCENES: Dict[str, SceneConfig] = {
    "campus": SceneConfig(max_obstacle_height_m=3.0, thicken_kernel_px=3, close_kernel_px=5),
    "forest": SceneConfig(max_obstacle_height_m=2.0, min_hits_per_cell=1, thicken_kernel_px=3, close_kernel_px=3),
    "garage": SceneConfig(max_obstacle_height_m=2.8, thicken_kernel_px=5, close_kernel_px=7),
    "indoor": SceneConfig(max_obstacle_height_m=2.6, thicken_kernel_px=5, close_kernel_px=7),
    "tunnel": SceneConfig(max_obstacle_height_m=2.4, thicken_kernel_px=5, close_kernel_px=7),
}


def read_binary_xyz_ply(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Incomplete PLY header: {path}")
            header_lines.append(line.decode("latin1").strip())
            if line.strip() == b"end_header":
                break

        if header_lines[0] != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        fmt = next((line for line in header_lines if line.startswith("format ")), "")
        if "binary_little_endian" not in fmt:
            raise ValueError(f"Unsupported PLY format for {path}: {fmt}")

        vertex_line = next((line for line in header_lines if line.startswith("element vertex ")), "")
        if not vertex_line:
            raise ValueError(f"Missing vertex count in {path}")
        vertex_count = int(vertex_line.split()[-1])

        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        points = np.fromfile(f, dtype=dtype, count=vertex_count)
        if points.size != vertex_count:
            raise ValueError(f"Unexpected vertex count in {path}: got {points.size}, expect {vertex_count}")
        return points


def estimate_ground_z(z_values: np.ndarray, cfg: SceneConfig) -> float:
    in_window = z_values[
        (z_values >= cfg.ground_hint_z - cfg.ground_search_half_window)
        & (z_values <= cfg.ground_hint_z + cfg.ground_search_half_window)
    ]
    candidate = in_window if in_window.size >= 1000 else z_values
    hist, edges = np.histogram(candidate, bins=256)
    mode_idx = int(hist.argmax())
    return float((edges[mode_idx] + edges[mode_idx + 1]) / 2.0)


def world_bounds(points: np.ndarray, margin_m: float) -> Tuple[float, float, float, float]:
    min_x = float(points["x"].min()) - margin_m
    max_x = float(points["x"].max()) + margin_m
    min_y = float(points["y"].min()) - margin_m
    max_y = float(points["y"].max()) + margin_m
    return min_x, max_x, min_y, max_y


def rasterize_obstacles(
    obstacle_points: np.ndarray,
    bounds: Tuple[float, float, float, float],
    resolution: float,
    min_hits_per_cell: int,
) -> np.ndarray:
    min_x, max_x, min_y, max_y = bounds
    width = int(math.ceil((max_x - min_x) / resolution))
    height = int(math.ceil((max_y - min_y) / resolution))

    x_idx = np.floor((obstacle_points["x"] - min_x) / resolution).astype(np.int32)
    y_idx = np.floor((max_y - obstacle_points["y"]) / resolution).astype(np.int32)

    valid = (x_idx >= 0) & (x_idx < width) & (y_idx >= 0) & (y_idx < height)
    x_idx = x_idx[valid]
    y_idx = y_idx[valid]

    counts = np.zeros((height, width), dtype=np.uint16)
    np.add.at(counts, (y_idx, x_idx), 1)
    return counts >= min_hits_per_cell


def apply_binary_cleanup(
    binary_obstacles: np.ndarray,
    thicken_kernel_px: int,
    close_kernel_px: int,
    open_kernel_px: int,
) -> np.ndarray:
    image = Image.fromarray((binary_obstacles.astype(np.uint8) * 255), mode="L")

    if thicken_kernel_px > 1:
        image = image.filter(ImageFilter.MaxFilter(size=thicken_kernel_px))

    if close_kernel_px > 1:
        image = image.filter(ImageFilter.MaxFilter(size=close_kernel_px))
        image = image.filter(ImageFilter.MinFilter(size=close_kernel_px))

    if open_kernel_px > 1:
        image = image.filter(ImageFilter.MinFilter(size=open_kernel_px))
        image = image.filter(ImageFilter.MaxFilter(size=open_kernel_px))

    return np.array(image) > 127


def save_outputs(scene_name: str, image_array: np.ndarray, metadata: Dict[str, object]) -> None:
    scene_output_dir = OUTPUT_ROOT / scene_name
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    image_path = scene_output_dir / "bev_clean.png"
    meta_path = scene_output_dir / "bev_clean_meta.json"

    Image.fromarray(image_array, mode="L").save(image_path)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[{scene_name}] saved image -> {image_path}")
    print(f"[{scene_name}] saved meta  -> {meta_path}")


def build_bev_for_scene(scene_name: str, cfg: SceneConfig) -> None:
    ply_path = MESH_ROOT / scene_name / "preview/pointcloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing pointcloud: {ply_path}")

    points = read_binary_xyz_ply(ply_path)
    ground_z = estimate_ground_z(points["z"], cfg)
    obstacle_min_z = ground_z + cfg.min_obstacle_height_m
    obstacle_max_z = ground_z + cfg.max_obstacle_height_m

    bounds = world_bounds(points, cfg.margin_m)
    obstacle_points = points[(points["z"] >= obstacle_min_z) & (points["z"] <= obstacle_max_z)]
    if obstacle_points.size == 0:
        raise ValueError(f"No obstacle points kept for scene {scene_name}")

    obstacle_mask = rasterize_obstacles(
        obstacle_points=obstacle_points,
        bounds=bounds,
        resolution=cfg.resolution,
        min_hits_per_cell=cfg.min_hits_per_cell,
    )
    obstacle_mask = apply_binary_cleanup(
        binary_obstacles=obstacle_mask,
        thicken_kernel_px=cfg.thicken_kernel_px,
        close_kernel_px=cfg.close_kernel_px,
        open_kernel_px=cfg.open_kernel_px,
    )

    bev = np.zeros(obstacle_mask.shape, dtype=np.uint8)
    bev[obstacle_mask] = 255

    min_x, max_x, min_y, max_y = bounds
    metadata = {
        "scene": scene_name,
        "source_pointcloud": str(ply_path),
        "image": "bev_clean.png",
        "semantics": {
            "free": 0,
            "obstacle": 255,
            "description": "black = navigable free space, white = obstacle",
        },
        "resolution_m_per_pixel": cfg.resolution,
        "origin_world": {
            "x_min": round(min_x, 6),
            "y_max": round(max_y, 6),
        },
        "bounds_world": {
            "x_min": round(min_x, 6),
            "x_max": round(max_x, 6),
            "y_min": round(min_y, 6),
            "y_max": round(max_y, 6),
        },
        "image_size": {
            "width": int(bev.shape[1]),
            "height": int(bev.shape[0]),
        },
        "ground_z": round(ground_z, 6),
        "obstacle_height_range": {
            "z_min": round(obstacle_min_z, 6),
            "z_max": round(obstacle_max_z, 6),
        },
        "rasterization": {
            "min_hits_per_cell": cfg.min_hits_per_cell,
            "thicken_kernel_px": cfg.thicken_kernel_px,
            "close_kernel_px": cfg.close_kernel_px,
            "open_kernel_px": cfg.open_kernel_px,
        },
        "pixel_to_world_note": "world_x = x_min + (px + 0.5) * resolution, world_y = y_max - (py + 0.5) * resolution",
    }

    print(
        f"[{scene_name}] ground_z={ground_z:.3f}, obstacle_points={obstacle_points.size}, "
        f"size={bev.shape[1]}x{bev.shape[0]}"
    )
    save_outputs(scene_name, bev, metadata)


def main(selected_scenes: Iterable[str]) -> None:
    for scene_name in selected_scenes:
        if scene_name not in SCENES:
            raise ValueError(f"Unsupported scene: {scene_name}")
        build_bev_for_scene(scene_name, SCENES[scene_name])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate clean binary BEV maps from preview pointclouds.")
    parser.add_argument("--mesh-root", type=Path, required=True,
                        help="Directory containing the original scene pointcloud resources")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT,
                        help="Output directory for generated maps and metadata")
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=list(SCENES.keys()),
        help="Subset of scenes to process. Default: all supported scenes.",
    )
    args = parser.parse_args()
    MESH_ROOT = args.mesh_root.resolve()
    OUTPUT_ROOT = args.output_root.resolve()
    main(args.scenes)
