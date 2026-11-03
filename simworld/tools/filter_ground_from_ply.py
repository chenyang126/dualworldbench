#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path

import numpy as np


def read_binary_xyz_ply(path: Path):
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Incomplete PLY header: {path}")
            header_lines.append(line.decode("latin1").rstrip("\n"))
            if line.strip() == b"end_header":
                break

        if not header_lines or header_lines[0] != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        fmt = next((line for line in header_lines if line.startswith("format ")), "")
        if "binary_little_endian" not in fmt:
            raise ValueError(f"Unsupported PLY format: {fmt}")

        vertex_line = next((line for line in header_lines if line.startswith("element vertex ")), "")
        if not vertex_line:
            raise ValueError(f"Missing vertex count in {path}")
        vertex_count = int(vertex_line.split()[-1])

        property_lines = [line for line in header_lines if line.startswith("property ")]
        xyz_prefix = [
            "property float x",
            "property float y",
            "property float z",
        ]
        if property_lines[:3] != xyz_prefix:
            raise ValueError("This tool expects the first 3 PLY properties to be float x/y/z")

        dtype_fields = []
        property_sizes = []
        property_names = []
        for line in property_lines:
            _, typ, name = line.split()
            property_names.append(name)
            if typ == "float":
                dtype_fields.append((name, "<f4"))
                property_sizes.append(4)
            elif typ == "uchar":
                dtype_fields.append((name, "u1"))
                property_sizes.append(1)
            else:
                raise ValueError(f"Unsupported PLY property type: {typ}")

        dtype = np.dtype(dtype_fields)
        points = np.fromfile(f, dtype=dtype, count=vertex_count)
        if points.size != vertex_count:
            raise ValueError(f"Unexpected vertex count in {path}: got {points.size}, expect {vertex_count}")
        return header_lines, points, property_names, property_sizes


def estimate_ground_z(z_values: np.ndarray, hint_z: float, half_window: float):
    in_window = z_values[(z_values >= hint_z - half_window) & (z_values <= hint_z + half_window)]
    candidate = in_window if in_window.size >= 1000 else z_values
    hist, edges = np.histogram(candidate, bins=256)
    mode_idx = int(hist.argmax())
    return float((edges[mode_idx] + edges[mode_idx + 1]) / 2.0)


def write_binary_ply(path: Path, header_lines, filtered_points):
    rewritten_header = []
    for line in header_lines:
        if line.startswith("element vertex "):
            rewritten_header.append(f"element vertex {len(filtered_points)}")
        else:
            rewritten_header.append(line)

    with path.open("wb") as f:
        for line in rewritten_header:
            f.write((line + "\n").encode("latin1"))
        filtered_points.tofile(f)


def main():
    parser = argparse.ArgumentParser(description="Filter near-ground points from a binary little-endian PLY map.")
    parser.add_argument("--input", required=True, help="Input PLY path")
    parser.add_argument("--output", required=True, help="Output filtered PLY path")
    parser.add_argument("--ground-hint-z", type=float, default=0.0, help="Ground height hint")
    parser.add_argument("--ground-search-half-window", type=float, default=1.5, help="Half window for ground search")
    parser.add_argument(
        "--ground-band-thickness",
        type=float,
        default=0.08,
        help="Remove points within [ground_z - t, ground_z + t]",
    )
    parser.add_argument(
        "--min-keep-z",
        type=float,
        default=None,
        help="Optional absolute minimum z to keep after ground filtering",
    )
    parser.add_argument(
        "--max-keep-z",
        type=float,
        default=None,
        help="Optional absolute maximum z to keep after ground filtering",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header_lines, points, _, _ = read_binary_xyz_ply(input_path)
    ground_z = estimate_ground_z(points["z"], args.ground_hint_z, args.ground_search_half_window)

    mask = np.abs(points["z"] - ground_z) > args.ground_band_thickness
    if args.min_keep_z is not None:
        mask &= points["z"] >= args.min_keep_z
    if args.max_keep_z is not None:
        mask &= points["z"] <= args.max_keep_z

    filtered_points = points[mask]
    write_binary_ply(output_path, header_lines, filtered_points)

    removed = int(points.size - filtered_points.size)
    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"ground_z={ground_z:.4f}")
    print(f"removed_points={removed}")
    print(f"kept_points={int(filtered_points.size)}")


if __name__ == "__main__":
    main()
