import math
import re
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

# --- Log Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("SimCore")

# --- 1. Utility Functions ---

def calculate_distance(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    return math.hypot(p1['x'] - p2['x'], p1['y'] - p2['y'])

def interpolate_pos(start: Dict[str, float], end: Dict[str, float], ratio: float) -> Dict[str, float]:
    return {
        "x": start['x'] + (end['x'] - start['x']) * ratio,
        "y": start['y'] + (end['y'] - start['y']) * ratio
    }

def extract_strict_json(text: str) -> Dict[str, Any]:
    """
    更加健壮的 JSON 提取工具，支持深层嵌套并过滤干扰字符。
    """
    if not text:
        return {}

    # 1. 预处理：移除 Markdown 标记
    clean_text = re.sub(r'```json\s*|\s*```', '', text).strip()

    # 2. 尝试寻找 JSON 的边界（处理嵌套）
    # 寻找第一个 { 和最后一个 }
    start_idx = clean_text.find('{')
    end_idx = clean_text.rfind('}')

    if start_idx == -1 or end_idx == -1:
        # 尝试寻找数组形式 [ ... ]
        start_idx = clean_text.find('[')
        end_idx = clean_text.rfind(']')
        
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        potential_json = clean_text[start_idx:end_idx + 1]
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            # 3. 容错处理：处理 LLM 常见的格式错误
            try:
                # 修复常见的 LLM 错误：如多余的逗号、错误的转义
                # 移除 JSON 字符串末尾对象后的多余逗号
                fixed_json = re.sub(r',\s*([\]}])', r'\1', potential_json)
                return json.loads(fixed_json)
            except Exception as e:
                logger.error(f"Deep parsing failed: {e}. Content: {potential_json[:500]}...")

    # 4. 终极回退：尝试全文本解析
    try:
        return json.loads(clean_text)
    except Exception as e:
        logger.error(f"Final fallback failed: {e}")
        # 打印前 200 个字符方便定位 LLM 哪里输错了
        logger.error(f"Raw text (first 1000 chars): {text[:1000]}")
        return {}

# --- 2. Planner Interface ---

class BasePlanner(ABC):
    @abstractmethod
    def plan(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Args:
            context: Contains speed, now, pos, max_capacity, carried, tasks, etc.
        Returns:
            JSON list of actions [{"action": "move", ...}, ...]
        """
        pass