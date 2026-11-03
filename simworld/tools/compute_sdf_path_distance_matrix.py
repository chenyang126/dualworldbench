#!/usr/bin/env python3
import argparse
import csv
import heapq
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path


POINT_RE = re.compile(r"\(([-0-9.]+),([-0-9.]+)\)")


def parse_points(path: Path):
    points = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            match = POINT_RE.fullmatch(text)
            if not match:
                raise ValueError(f"Invalid point format at line {idx}: {text}")
            points.append((float(match.group(1)), float(match.group(2))))
    if not points:
        raise ValueError(f"No points found in {path}")
    return points


def parse_pose(text: str):
    values = [float(part) for part in text.split()]
    if len(values) != 6:
        raise ValueError(f"Unexpected pose format: {text}")
    return values


def parse_sdf_rectangles(path: Path):
    root = ET.parse(path).getroot()
    floor_rect = None
    obstacle_rects = []

    for collision in root.findall(".//collision"):
        name = collision.attrib.get("name", "")
        size_text = collision.findtext("./geometry/box/size")
        if size_text is None:
            continue
        sx, sy, sz = [float(part) for part in size_text.split()]
        pose_text = collision.findtext("./pose", default="0 0 0 0 0 0")
        px, py, pz, roll, pitch, yaw = parse_pose(pose_text)
        if abs(yaw) > 1e-6 or abs(roll) > 1e-6 or abs(pitch) > 1e-6:
            raise ValueError(f"Only axis-aligned boxes are supported: {name}")

        rect = {
            "name": name,
            "cx": px,
            "cy": py,
            "sx": sx,
            "sy": sy,
            "xmin": px - sx / 2.0,
            "xmax": px + sx / 2.0,
            "ymin": py - sy / 2.0,
            "ymax": py + sy / 2.0,
        }

        if name == "floor_collision":
            floor_rect = rect
        else:
            obstacle_rects.append(rect)

    if floor_rect is None:
        raise ValueError(f"Could not find floor_collision in {path}")
    return floor_rect, obstacle_rects


class OccupancyGrid:
    def __init__(self, floor_rect, obstacle_rects, resolution: float, inflation: float):
        self.resolution = resolution
        self.inflation = inflation
        self.xmin = floor_rect["xmin"]
        self.xmax = floor_rect["xmax"]
        self.ymin = floor_rect["ymin"]
        self.ymax = floor_rect["ymax"]
        self.width = int(math.ceil((self.xmax - self.xmin) / resolution))
        self.height = int(math.ceil((self.ymax - self.ymin) / resolution))
        self.blocked = bytearray(self.width * self.height)

        for iy in range(self.height):
            y = self.cell_center_y(iy)
            for ix in range(self.width):
                x = self.cell_center_x(ix)
                if self._is_blocked(x, y, obstacle_rects):
                    self.blocked[self.index(ix, iy)] = 1

    def _is_blocked(self, x, y, obstacle_rects):
        for rect in obstacle_rects:
            if (
                rect["xmin"] - self.inflation <= x <= rect["xmax"] + self.inflation
                and rect["ymin"] - self.inflation <= y <= rect["ymax"] + self.inflation
            ):
                return True
        return False

    def index(self, ix, iy):
        return iy * self.width + ix

    def in_bounds(self, ix, iy):
        return 0 <= ix < self.width and 0 <= iy < self.height

    def is_free(self, ix, iy):
        return self.in_bounds(ix, iy) and self.blocked[self.index(ix, iy)] == 0

    def cell_center_x(self, ix):
        return self.xmin + (ix + 0.5) * self.resolution

    def cell_center_y(self, iy):
        return self.ymin + (iy + 0.5) * self.resolution

    def point_to_cell(self, point):
        x, y = point
        ix = int(math.floor((x - self.xmin) / self.resolution))
        iy = int(math.floor((y - self.ymin) / self.resolution))
        if not self.in_bounds(ix, iy):
            raise ValueError(f"Point {point} is outside the floor bounds")
        if self.is_free(ix, iy):
            return ix, iy
        return self.find_nearest_free(ix, iy)

    def find_nearest_free(self, ix, iy, max_radius_cells=20):
        if self.is_free(ix, iy):
            return ix, iy
        for radius in range(1, max_radius_cells + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    nx = ix + dx
                    ny = iy + dy
                    if self.is_free(nx, ny):
                        return nx, ny
        raise ValueError(f"No nearby free cell found around blocked point cell ({ix}, {iy})")

    def dijkstra_from(self, start_cell):
        start_idx = self.index(*start_cell)
        dist = {start_idx: 0.0}
        heap = [(0.0, start_cell[0], start_cell[1])]
        neighbors = [
            (-1, 0, self.resolution),
            (1, 0, self.resolution),
            (0, -1, self.resolution),
            (0, 1, self.resolution),
            (-1, -1, self.resolution * math.sqrt(2.0)),
            (-1, 1, self.resolution * math.sqrt(2.0)),
            (1, -1, self.resolution * math.sqrt(2.0)),
            (1, 1, self.resolution * math.sqrt(2.0)),
        ]

        while heap:
            cur_dist, ix, iy = heapq.heappop(heap)
            cur_idx = self.index(ix, iy)
            if cur_dist > dist[cur_idx]:
                continue
            for dx, dy, step_cost in neighbors:
                nx = ix + dx
                ny = iy + dy
                if not self.is_free(nx, ny):
                    continue
                nxt_idx = self.index(nx, ny)
                cand = cur_dist + step_cost
                if cand < dist.get(nxt_idx, float("inf")):
                    dist[nxt_idx] = cand
                    heapq.heappush(heap, (cand, nx, ny))
        return dist


def write_matrix_csv(path: Path, headers, matrix):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point"] + headers)
        for label, row in zip(headers, matrix):
            writer.writerow([label] + [f"{value:.3f}" if math.isfinite(value) else "inf" for value in row])


def write_mapping_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "point_id",
                "point_x",
                "point_y",
                "cell_x",
                "cell_y",
                "snap_x",
                "snap_y",
                "snap_distance_m",
            ]
        )
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute pairwise traversable-path distances from an SDF scene."
    )
    parser.add_argument("--points", required=True, help="Point list file with lines formatted as (x,y)")
    parser.add_argument("--sdf", required=True, help="Scene model.sdf path")
    parser.add_argument("--distance-output", required=True, help="Output CSV path")
    parser.add_argument("--mapping-output", default=None, help="Optional point snap mapping CSV")
    parser.add_argument("--resolution", type=float, default=0.1, help="Occupancy grid resolution in meters")
    parser.add_argument("--inflation", type=float, default=0.3, help="Obstacle inflation radius in meters")
    args = parser.parse_args()

    points = parse_points(Path(args.points))
    floor_rect, obstacle_rects = parse_sdf_rectangles(Path(args.sdf))
    grid = OccupancyGrid(
        floor_rect=floor_rect,
        obstacle_rects=obstacle_rects,
        resolution=args.resolution,
        inflation=args.inflation,
    )

    labels = [f"P{i:02d}" for i in range(1, len(points) + 1)]
    snapped_cells = []
    mapping_rows = []
    for idx, point in enumerate(points, start=1):
        ix, iy = grid.point_to_cell(point)
        snap_x = grid.cell_center_x(ix)
        snap_y = grid.cell_center_y(iy)
        snap_dist = math.hypot(point[0] - snap_x, point[1] - snap_y)
        snapped_cells.append((ix, iy, snap_dist))
        mapping_rows.append(
            [
                idx,
                f"{point[0]:.3f}",
                f"{point[1]:.3f}",
                ix,
                iy,
                f"{snap_x:.3f}",
                f"{snap_y:.3f}",
                f"{snap_dist:.3f}",
            ]
        )

    cell_cache = {}
    matrix = []
    for row_idx, (ix, iy, start_snap) in enumerate(snapped_cells):
        start_key = (ix, iy)
        if start_key not in cell_cache:
            cell_cache[start_key] = grid.dijkstra_from((ix, iy))
        dist_map = cell_cache[start_key]
        row = []
        for col_idx, (jx, jy, end_snap) in enumerate(snapped_cells):
            if row_idx == col_idx:
                row.append(0.0)
                continue
            target_idx = grid.index(jx, jy)
            path_dist = dist_map.get(target_idx, float("inf"))
            if math.isfinite(path_dist):
                row.append(start_snap + path_dist + end_snap)
            else:
                row.append(float("inf"))
        matrix.append(row)

    distance_output = Path(args.distance_output)
    distance_output.parent.mkdir(parents=True, exist_ok=True)
    write_matrix_csv(distance_output, labels, matrix)

    if args.mapping_output:
        mapping_output = Path(args.mapping_output)
        mapping_output.parent.mkdir(parents=True, exist_ok=True)
        write_mapping_csv(mapping_output, mapping_rows)


if __name__ == "__main__":
    main()
