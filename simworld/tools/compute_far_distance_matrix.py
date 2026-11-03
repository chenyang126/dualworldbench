#!/usr/bin/env python3
import argparse
import csv
import heapq
import math
import re
from pathlib import Path


POINT_RE = re.compile(r"\(([-0-9.]+),([-0-9.]+)\)")


def parse_points(path: Path):
    points = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            match = POINT_RE.fullmatch(line)
            if not match:
                raise ValueError(f"Invalid point format at line {idx}: {line}")
            points.append((float(match.group(1)), float(match.group(2))))
    return points


def parse_vgh(path: Path):
    nodes = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            components = line.split()
            if len(components) < 15:
                continue

            node_id = int(components[0])
            x = float(components[2])
            y = float(components[3])
            is_navpoint = bool(int(components[13]))

            connect_ids = []
            for token in components[15:]:
                if token == "|":
                    break
                connect_ids.append(int(token))

            nodes[node_id] = {
                "id": node_id,
                "x": x,
                "y": y,
                "is_navpoint": is_navpoint,
                "connect_ids": connect_ids,
            }
    return nodes


def build_graph(nodes):
    graph = {node_id: {} for node_id in nodes}
    for node_id, node in nodes.items():
        x1, y1 = node["x"], node["y"]
        for neighbor_id in node["connect_ids"]:
            if neighbor_id not in nodes:
                continue
            x2, y2 = nodes[neighbor_id]["x"], nodes[neighbor_id]["y"]
            dist = math.hypot(x1 - x2, y1 - y2)
            graph[node_id][neighbor_id] = dist
            graph[neighbor_id][node_id] = dist
    return graph


def nearest_navpoint(point, nodes):
    px, py = point
    best = None
    best_dist = float("inf")
    for node in nodes.values():
        if not node["is_navpoint"]:
            continue
        dist = math.hypot(node["x"] - px, node["y"] - py)
        if dist < best_dist:
            best = node
            best_dist = dist
    if best is None:
        raise ValueError("No navpoint found in VGH")
    return best, best_dist


def dijkstra(graph, start_id):
    dist = {start_id: 0.0}
    heap = [(0.0, start_id)]
    while heap:
        cur_dist, cur_id = heapq.heappop(heap)
        if cur_dist > dist[cur_id]:
            continue
        for nxt_id, weight in graph[cur_id].items():
            cand = cur_dist + weight
            if cand < dist.get(nxt_id, float("inf")):
                dist[nxt_id] = cand
                heapq.heappush(heap, (cand, nxt_id))
    return dist


def write_matrix_csv(path: Path, headers, matrix):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point"] + headers)
        for label, row in zip(headers, matrix):
            writer.writerow([label] + [f"{value:.3f}" for value in row])


def write_mapping_csv(path: Path, mapping_rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "point_id",
                "point_x",
                "point_y",
                "nearest_nav_id",
                "nav_x",
                "nav_y",
                "access_distance_m",
            ]
        )
        writer.writerows(mapping_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute a FAR-planner distance matrix from points and a VGH graph."
    )
    parser.add_argument(
        "--points",
        required=True,
        help="Point list file, one point per line, format: (x,y)",
    )
    parser.add_argument(
        "--vgh",
        required=True,
        help="FAR planner .vgh graph file",
    )
    parser.add_argument(
        "--distance-output",
        required=True,
        help="Output CSV path for the pairwise distance matrix",
    )
    parser.add_argument(
        "--mapping-output",
        default=None,
        help="Optional output CSV path for point-to-navpoint mapping",
    )
    args = parser.parse_args()

    points_path = Path(args.points)
    vgh_path = Path(args.vgh)
    distance_output = Path(args.distance_output)
    mapping_output = Path(args.mapping_output) if args.mapping_output else None

    distance_output.parent.mkdir(parents=True, exist_ok=True)
    if mapping_output:
        mapping_output.parent.mkdir(parents=True, exist_ok=True)

    points = parse_points(points_path)
    nodes = parse_vgh(vgh_path)
    graph = build_graph(nodes)

    snapped = []
    mapping_rows = []
    for idx, point in enumerate(points, start=1):
        nav_node, access_dist = nearest_navpoint(point, nodes)
        snapped.append((nav_node["id"], access_dist))
        mapping_rows.append(
            [
                idx,
                f"{point[0]:.3f}",
                f"{point[1]:.3f}",
                nav_node["id"],
                f"{nav_node['x']:.3f}",
                f"{nav_node['y']:.3f}",
                f"{access_dist:.3f}",
            ]
        )

    labels = [f"P{i:02d}" for i in range(1, len(points) + 1)]
    all_pairs_dist = []
    cache = {}
    for i, (start_node_id, start_access) in enumerate(snapped):
        if start_node_id not in cache:
            cache[start_node_id] = dijkstra(graph, start_node_id)
        shortest = cache[start_node_id]
        row = []
        for end_node_id, end_access in snapped:
            graph_dist = shortest.get(end_node_id, float("inf"))
            if math.isinf(graph_dist):
                row.append(float("inf"))
            else:
                row.append(start_access + graph_dist + end_access)
        row[i] = 0.0
        all_pairs_dist.append(row)

    write_matrix_csv(distance_output, labels, all_pairs_dist)
    if mapping_output:
        write_mapping_csv(mapping_output, mapping_rows)


if __name__ == "__main__":
    main()
