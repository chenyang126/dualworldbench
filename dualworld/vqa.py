"""DualWorld-VQA loading, canonical model inputs and offline judges."""

from ._paths import bootstrap

bootstrap()
from run_vqa_api import (resolve_dataset_paths, load_dataset,
                         build_messages, evaluate_record)


def load_questions(sets=("l1", "l2", "l3"), scenes=None):
    """Return evaluation records with resolved image paths; deduplicate IDs.

    Records include annotations. Only build_messages(record) should be sent
    to a model; never expose raw records or ground truth as model input.
    """
    if isinstance(sets, str):
        sets = (sets,)
    scene_filter = set(scenes) if scenes is not None else None
    records = {}
    for path in resolve_dataset_paths(sets):
        for record in load_dataset(path, None, scene_filter, None):
            records[record["question_id"]] = record
    return list(records.values())


def judge(record, answer):
    """Return the same result as the API runner's offline judge."""
    return evaluate_record(record, str(answer))
