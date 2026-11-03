from __future__ import annotations

import json
import os
import logging
import time
from typing import Any, Dict, List, Type

from llm_client import create_openai_client, temperature_kwargs
from node_sequence import parse_node_sequence
from planner_base import BasePlanner
from prompt_utils import (
    build_mapping_lines,
    build_multimodal_user_content,
    build_previous_execution_prompt,
    summarize_tasks_for_prompt,
)
from prompt_registry import get_planner_prompt


logger = logging.getLogger("LLMPlanner")

PREVIOUS_EXECUTION_HEADING = "PREVIOUS EXECUTION RESULT"


PLAN_CALL_TIMEOUT_S = float(os.getenv("PLAN_CALL_TIMEOUT_S", "1800"))


class BaseLLMPlanner(BasePlanner):
    """Planner that emits the shared node sequence, whatever its input modality.

    Every modality is asked for the same simplified output format, so the only
    variable across meta / text / image is the information they receive. The node
    sequence is expanded into an executable point sequence at evaluation time.
    """

    planner_type = "meta"
    require_image = False
    # Set by plan(): the transport-level failure behind an empty result, or None
    # when the call succeeded and the model itself produced nothing usable.
    last_error: str | None = None
    # Raw model body from the most recent call, so a caller recording an
    # unparseable result can store the evidence instead of only a status.
    last_raw: str | None = None

    def __init__(self, model: str = "gpt-4o", scene: str = "indoor", planner_name: str = "LLMPlanner"):
        self.scene = scene
        # base_url=None defers to llm_client.DEFAULT_BASE_URL, which honors
        # OPENAI_BASE_URL so the endpoint can be switched without a code edit.
        self.client, self.model = create_openai_client(
            model=model,
            planner_name=planner_name,
            base_url=None,
        )
        self.prompt_template = get_planner_prompt(scene, self.planner_type)

    def prompt_fields(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Values substituted into the scene prompt template."""
        return {}

    def build_prompt(self, context: Dict[str, Any]) -> str:
        prompt = self.prompt_template.format(**self.prompt_fields(context))
        return prompt + build_previous_execution_prompt(
            context.get("prev_execution"),
            heading=PREVIOUS_EXECUTION_HEADING,
        )

    def build_messages(self, prompt: str, context: Dict[str, Any]) -> Any:
        if self.require_image:
            return [{"role": "user", "content": build_multimodal_user_content(prompt, context, require_image=True)}]
        return [{"role": "user", "content": prompt}]

    def parse_response(self, content: str, context: Dict[str, Any]) -> List[str]:
        return parse_node_sequence(content)

    def plan(self, context: Dict[str, Any]) -> List[str]:
        # An empty return means two very different things: the transport failed
        # (retry it) or the model answered something unparseable (a genuine model
        # failure that must be scored, not retried). Record which, so a caller
        # running a large sweep does not charge rate limiting to the model.
        self.last_error: str | None = None
        self.last_raw = None
        prompt = self.build_prompt(context)
        try:
            messages = self.build_messages(prompt, context)
            time.sleep(3)
            print(f"正在调用 GPT API (model: {self.model}, scene: {self.scene}, planner: {self.planner_type})...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                # Reasoning models need far longer than the fast ones on the
                # text prompt: qwen3.7-plus/text has a median of ~865s and a
                # p90 of ~2750s, against 135-230s for image and multimodal.
                # At the old 300s ceiling those calls could only ever time out,
                # burning every retry and recording an api_error for a cell the
                # model can actually solve. Raising the ceiling cannot change a
                # completed answer, only whether we wait long enough to get one.
                timeout=PLAN_CALL_TIMEOUT_S,
                **temperature_kwargs(self.model, 0.0),
            )
            if not response or not response.choices:
                logger.error("GPT API 返回空响应")
                self.last_error = "empty_response"
                return []

            content = response.choices[0].message.content or ""
            self.last_raw = content
            # An empty body is a transport failure, not an answer. The gateway
            # intermittently returns finish_reason=stop with zero-length content
            # (and output_tokens=0); the identical cell succeeds on a retry.
            # Classifying it as "model produced nothing parseable" would make
            # run_job() refuse to retry and score infrastructure noise against
            # the model, so flag it as retryable instead.
            if not content.strip():
                logger.error("GPT API 返回空 content（视为传输失败，可重试）")
                self.last_error = "empty_content"
                return []
            print(f"收到响应，长度: {len(content)}")
            node_sequence = self.parse_response(content, context)
            if not node_sequence:
                logger.error("无法从模型输出中解析有效节点序列")
                print(f"Raw Content: {content}")
            else:
                print(f"成功解析 {len(node_sequence)} 个节点: {' -> '.join(node_sequence)}")
            return node_sequence
        except Exception as exc:
            logger.error(f"{self.__class__.__name__} failed: {exc}", exc_info=True)
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []


class ActionPlanner(BaseLLMPlanner):
    planner_type = "meta"
    require_image = True
    include_coords = False

    def prompt_fields(self, context: Dict[str, Any]) -> Dict[str, Any]:
        carried_tasks = summarize_tasks_for_prompt(context.get("carried", []), include_coords=self.include_coords)
        available_tasks = summarize_tasks_for_prompt(context.get("tasks", []), include_coords=self.include_coords)
        mapping_text = "\n".join(build_mapping_lines(context)) or "No active tasks."
        return {
            "now": context["now"],
            "cur_x": context["pos"]["x"],
            "cur_y": context["pos"]["y"],
            "speed": context["speed"],
            "capacity": len(context.get("carried", [])),
            "max_capacity": context["max_capacity"],
            "carried_tasks_json": json.dumps(carried_tasks, ensure_ascii=False),
            "new_tasks_json": json.dumps(available_tasks, ensure_ascii=False),
            "id_mapping_section": mapping_text,
        }


class TextPlanner(ActionPlanner):
    planner_type = "text"
    require_image = False
    include_coords = True


class ImagePlanner(BaseLLMPlanner):
    planner_type = "image"
    require_image = True


PLANNER_CLASSES: Dict[str, Type[BaseLLMPlanner]] = {
    "meta": ActionPlanner,
    "text": TextPlanner,
    "image": ImagePlanner,
}


def get_planner_class(planner_type: str) -> Type[BaseLLMPlanner]:
    try:
        return PLANNER_CLASSES[planner_type]
    except KeyError as exc:
        raise ValueError(f"不支持的 planner_type: {planner_type}") from exc
