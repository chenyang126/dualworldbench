"""SimWorld data, prompt and offline scoring interfaces."""

import re
from pathlib import Path

from ._paths import bootstrap

bootstrap()
from grid import MODALITIES, available_scenes, keep_sample
from run_simworld import load_context
from run_planner import list_sample_ids
from scene_config import get_scene_config
from planner import get_planner_class
from prompt_registry import get_planner_prompt
from score_simworld import read_predictions, score_predictions


def list_scenes():
    return available_scenes()


def list_samples(scene):
    config = get_scene_config(scene)
    return [sample for sample in list_sample_ids(config.default_task_dir())
            if keep_sample(scene, sample)]


def load_task(scene, sample_id, modality="meta"):
    """Return normalized evaluator context; modality: text/image/meta."""
    if modality not in MODALITIES:
        raise ValueError(f"Unknown modality: {modality}")
    if not re.fullmatch(r"n\d+-\d+", sample_id) or sample_id not in list_samples(scene):
        raise ValueError(f"Sample outside the standard grid: {scene}/{sample_id}")
    return load_context(scene, sample_id, modality)


def build_messages(scene, sample_id, modality="meta"):
    """Build the canonical model input without instantiating an API client."""
    context = load_task(scene, sample_id, modality)
    cls = get_planner_class(modality)
    planner = cls.__new__(cls)
    planner.scene = scene
    planner.prompt_template = get_planner_prompt(scene, modality)
    return planner.build_messages(planner.build_prompt(context), context)


def score_file(predictions, scenes=None, modalities=None, allow_partial=False,
               reference_file=None):
    """Replay a JSONL prediction file and return metrics and coverage."""
    return score_predictions(read_predictions(Path(predictions)),
                             list_scenes() if scenes is None else scenes,
                             list(MODALITIES) if modalities is None else modalities,
                             allow_partial, reference_file)
