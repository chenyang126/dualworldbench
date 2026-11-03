import json
import logging
import os
import sys
import time
from typing import Any, Dict, List
from common import BasePlanner, extract_strict_json
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from model_endpoints import resolve as resolve_endpoint, temperature_kwargs

logger = logging.getLogger("GPTPlanner")

# --- 核心 Prompt (注入物理法则 & 保持接口一致) ---
PLANNER_PROMPT = """You are the logistics control brain for a delivery robot.
Generate a strictly sequential action plan to maximize total revenue.

### 1. CURRENT STATE
- **Time:** {now:.2f} s
- **Position:** {{ "x": {cur_x:.2f}, "y": {cur_y:.2f} }}
- **Speed:** {speed:.2f} m/s
- **Load:** {capacity}/{max_capacity}
{interrupt_section}

### 2. TASK POOL
**A. Carried Tasks (Deliver Only):**
{carried_tasks_json}

**B. Available Tasks (Pickup -> Deliver):**
{new_tasks_json}

### 3. PHYSICS & CALCULATION RULES (STRICT)
1. **Euclidean Distance:**
   - Distance between (x1, y1) and (x2, y2) = `sqrt((x2 - x1)^2 + (y2 - y1)^2)`
2. **Travel Time:**
   - `Duration = Distance / Speed` (Current Speed: {speed:.2f} m/s)
   - `Arrival Time = Previous Time + Duration`
3. **Movement:**
   - You MUST `move` to the exact target coordinates before `pickup` or `deliver`.
   - Action Sequence: `move` -> `pickup` -> `move` -> `deliver`.

### 4. BUSINESS CONSTRAINTS
1. **Capacity:** Do not pickup if `current_load` >= {max_capacity}.
2. **Priority (Hard Constraint):** If carrying **Type C**, deliver it BEFORE A or B.
3. **Revenue Calculation:** `revenue = max(0, reward - (arrival_time - deadline) * penalty_rate)`.
   - Penalty rates: A=0.1, B=0.2, C=0.3 yuan/s.
   - Note: `reward` is the base reward from task data, `revenue` is the calculated actual income after penalties.

### 5. OUTPUT FORMAT
Return **ONLY** a raw JSON object with a single key "actions". Do not include any preamble, postamble, or markdown blocks.”
The format must match the simulator's input requirements exactly.

**Required JSON Structure:**
{{
  "actions": [
    {{
      "type": "move",
      "to": {{ "x": float, "y": float }}  // Target coordinates
      "time": float
    }},
    {{
      "type": "pickup",
      "id": "string",
      "time": float
    }},
    {{
      "type": "deliver",
      "id": "string",
      "time": float,
      "revenue": float
    }}
  ]
}}
"""

class GPTPlanner(BasePlanner):
    def __init__(self, model="gpt-4o"):
        # Use the user-configured OpenAI-compatible endpoint.
        base_url, api_key = resolve_endpoint(model)
        # 增加 max_retries 到 10 以应对频繁的 429 错误
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=10)

        self.model = model

    def plan(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        # 1. 数据整理
        carried_list = []
        for t in context['carried']:
            carried_list.append({
                "id": t['id'],
                "type": t['type'],
                "deliver": t['deliver'],
                "deadline": t['end_time'],
                "reward": t['reward']
            })

        new_list = []
        for t in context['tasks']:
            new_list.append({
                "id": t['id'],
                "type": t['type'],
                "pickup": t['pickup'],
                "deliver": t['deliver'],
                "deadline": t['end_time'],
                "reward": t['reward']
            })

        # 2. 中断处理
        interrupt_section = ""
        if context.get('last_interrupt'):
            interrupt_section = "- **NOTE:** Previous move INTERRUPTED. Plan strictly from CURRENT Position."

        # 3. 构建 Prompt
        prompt = PLANNER_PROMPT.format(
            now=context['now'],
            cur_x=context['pos']['x'],
            cur_y=context['pos']['y'],
            speed=context['speed'],
            capacity=len(context['carried']),
            max_capacity=context['max_capacity'],
            interrupt_section=interrupt_section,
            carried_tasks_json=json.dumps(carried_list, ensure_ascii=False),
            new_tasks_json=json.dumps(new_list, ensure_ascii=False)
        )

        # 4. 调用 LLM
        # 4. 调用 LLM
        content = None
        max_retries = 8
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                # 初始微小延迟，后续重试使用指数退避
                if attempt == 0:
                    time.sleep(2)
                
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=float(os.getenv("TEXTWORLD_API_TIMEOUT", "600")),
                    **temperature_kwargs(self.model, 0.0),
                )
                
                content = resp.choices[0].message.content or ""
                # An empty body is a gateway hiccup, not a decision to do
                # nothing. Returning [] here would be scored as the agent
                # idling for the whole replan window, so retry it the same way
                # a 429 is retried.
                if not content.strip():
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt)
                        logger.warning(
                            f"[empty content] Retry {attempt+1}/{max_retries} after {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        content = None
                        continue
                    logger.error("GPT returned empty content on every attempt")
                    return []
                break # 成功则跳出重试循环

            except Exception as e:
                error_str = str(e)
                # 针对 429 Rate Limit 和配额不足进行重试
                if "429" in error_str or "insufficient_quota" in error_str or "Too Many Requests" in error_str or "Rate limit" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt)
                        logger.warning(f"[429 Rate Limit] Retry {attempt+1}/{max_retries} after {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                
                # 其他错误或重试耗尽
                logger.error(f"GPT Planning Failed: {e}")
                return []

        if not content:
            return []

        try:
            data = extract_strict_json(content)
            actions = data.get("actions", [])
            
            # 5. 格式兼容性修正 (Safety Check)
            # 虽然 Prompt 已经要求很严，但为了防止 simulator 报错，这里做一层清洗
            clean_actions = []
            for act in actions:
                # 直接读取 type 字段
                act_type = act.get('type')
                
                if not act_type: 
                    continue
                
                # 根据不同的动作类型，读取所有相应的字段
                new_act = {"action": act_type}
                
                if act_type == 'move':
                    # move 动作：读取 type, to, time
                    new_act['to'] = act.get('to')
                    new_act['time'] = act.get('time')
                elif act_type == 'pickup':
                    # pickup 动作：读取 type, id, time
                    new_act['task_id'] = act.get('id')
                    new_act['time'] = act.get('time')
                elif act_type == 'deliver':
                    # deliver 动作：读取 type, id, time, revenue（模型计算的收益）
                    new_act['task_id'] = act.get('id')
                    new_act['time'] = act.get('time')
                    # 保存模型输出的 revenue（模型计算的收益，用于对比 simulator 计算的 revenue）
                    llm_revenue = act.get('revenue')
                    if llm_revenue is not None:
                        new_act['revenue'] = llm_revenue
                
                clean_actions.append(new_act)

            return clean_actions

        except Exception as e:
            logger.error(f"Result Parsing Failed: {e}")
            return []
