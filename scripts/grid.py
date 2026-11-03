#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The canonical evaluation grid, shared by the runner and the analyzer.

Keeping this in one module is not cosmetic: if the runner's job list and the
analyzer's cell set disagree, the analyzer silently averages over a different
population than the one that was run, and nothing errors.

Every scene is a 3 x 5 design: task count n in {5, 8, 10} x 5 instances. Twelve
scenes are complete at 15 samples. `typical` (the Open column) is missing
n8-4, n8-5, n10-4 and n10-5 on disk, so taking all of its files would give
11 samples skewed toward the smallest instances (n5:n8:n10 = 5:3:3 rather than
5:5:5). Open is therefore restricted to instances 1-3 of each task count: 9
samples, balanced 3:3:3, comparable in composition to every other column.
"""

from __future__ import annotations

import os
from typing import Dict, List, Set

TASK_COUNTS = (5, 8, 10)
MODALITIES = ("meta", "text", "image")
MODELS = (
    "gpt-5.6-sol",
    "gpt-5.4-mini",
    "claude-opus-5",
    "claude-haiku-4-5-20251001",
    "qwen3.7-plus",
    "qwen3-vl-8b-instruct",
)

# Scenes whose sample set is deliberately narrower than what the directory holds.
SAMPLE_ALLOWLIST: Dict[str, Set[str]] = {
    "typical": {f"n{n}-{k}" for n in TASK_COUNTS for k in (1, 2, 3)},
}


def keep_sample(scene: str, sample_id: str) -> bool:
    allow = SAMPLE_ALLOWLIST.get(scene)
    return sample_id in allow if allow is not None else True


def select_samples(scene: str, sample_ids: List[str]) -> List[str]:
    return [s for s in sample_ids if keep_sample(scene, s)]


GROUP_OF_SINGLE = {"typical": "Open", "indoor": "Indoor", "campus": "Campus"}
GROUPS = ("Open", "T1", "T2", "Indoor", "Campus")


def group_of(scene: str) -> str:
    if scene in GROUP_OF_SINGLE:
        return GROUP_OF_SINGLE[scene]
    if scene.startswith("l1"):
        return "T1"
    if scene.startswith("l2"):
        return "T2"
    return "Other"


def available_scenes() -> List[str]:
    from scene_config import SCENE_CONFIGS, get_scene_config

    return [
        s for s in sorted(SCENE_CONFIGS)
        if os.path.isdir(get_scene_config(s).default_task_dir())
    ]
