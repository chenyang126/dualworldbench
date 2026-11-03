#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Literal, Tuple
import random


TaskType = Literal["A", "B", "C"]


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Task:
    id: str
    type: TaskType
    type_name: str
    title: str
    trigger_time: float
    pickup: Point
    deliver: Point
    distance_m: float
    end_time: float
    reward: float
    priority: bool


TYPE_NAME_MAP: Dict[TaskType, str] = {
    "A": "Standard Delivery",
    "B": "Express Delivery",
    "C": "Direct Delivery",
}


TYPE_ITEM_TITLES: Dict[TaskType, List[str]] = {
    "A": ["Milk Tea", "Coffee", "Bread", "Fresh Food"],
    "B": ["Cake", "Medicine", "Digital Accessories", "Documents"],
    "C": ["Flowers", "Phone", "Contract", "Keys"],
}

# 根据给定的权重比例和总任务数目，计算出三类任务各自的确切数量
def generate_distribution(total: int, weights: Tuple[int, int, int]) -> List[TaskType]:
    """Generate an exact count list of task types matching the 7:2:1 ratio (or custom).

    Uses largest-remainder method to match the total exactly.
    """
    w_a, w_b, w_c = weights
    weight_sum = w_a + w_b + w_c
    exact_a = total * w_a / weight_sum
    exact_b = total * w_b / weight_sum
    exact_c = total * w_c / weight_sum

    floor_a = math.floor(exact_a)
    floor_b = math.floor(exact_b)
    floor_c = math.floor(exact_c)
    assigned = floor_a + floor_b + floor_c
    remainder = total - assigned

    # 构建仅包含正权重类型的余数列表，避免把剩余量分配到权重为0的类型
    rem_items: List[Tuple[TaskType, float]] = []
    if w_a > 0:
        rem_items.append(("A", exact_a - floor_a))
    if w_b > 0:
        rem_items.append(("B", exact_b - floor_b))
    if w_c > 0:
        rem_items.append(("C", exact_c - floor_c))
    # 若全部为0（理论不应发生），退化为全A
    if not rem_items:
        rem_items = [("A", 1.0)]

    # Distribute remaining counts to the largest fractional remainders
    rem_items.sort(key=lambda x: x[1], reverse=True)
    add = {"A": 0, "B": 0, "C": 0}
    for i in range(remainder):
        add[rem_items[i % len(rem_items)][0]] += 1

    count_a = floor_a + add["A"]
    count_b = floor_b + add["B"]
    count_c = floor_c + add["C"]

    result: List[TaskType] = ["A"] * count_a + ["B"] * count_b + ["C"] * count_c
    return result

# 计算两点之间的欧几里得距离
def euclidean_distance(a: Point, b: Point) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return math.hypot(dx, dy)

# 价值计算函数，根据距离分段计算对应的值
# def step_value(distance_m: float, base_threshold_m: int, base_value: float, step_size_m: int, step_increment: float) -> float:
#     """Compute a step-wise value by distance.

#     - Within base_threshold_m (inclusive) returns base_value.
#     - Beyond threshold, each step_size_m adds step_increment (ceil for any remainder beyond a full step).
#     """
#     if distance_m <= base_threshold_m:
#         return base_value
#     extra_m = distance_m - base_threshold_m
#     steps = math.ceil(extra_m / step_size_m)
#     return base_value + steps * step_increment

# 格式化金额，保留两位小数
def format_amount(amount: float) -> float:
    # Round to 2 decimals, avoid representation noise
    return float(f"{amount:.2f}")

def get_task_params(task_type: TaskType, rng: random.Random) -> Tuple[float, float]:
    # Returns (reward, time_factor)
    if task_type == "A":
        reward = rng.uniform(10, 30)
        factor = 5.0
    elif task_type == "B":
        reward = rng.uniform(30, 60)
        factor = 3.0  
    else: # C
        reward = rng.uniform(60, 100)
        factor = 2.0  
    return reward, factor

# 构建单个任务对象
def build_task(
    idx: int,
    task_type: TaskType,
    trigger_time: float,
    pickup: Point,
    deliver: Point,
    rng: random.Random,
    level: int,
    speed: float
) -> Task:
    distance_m = euclidean_distance(pickup, deliver)
    
    # 根据起点到终点的距离计算预估时间
    # speed is m/min. Convert to seconds.
    estimated_time_sec = (distance_m / speed) * 60
    
    reward_val, time_factor = get_task_params(task_type, rng)
    
    # 如果level为1，则end_time设为无限
    if level == 1:
        end_time = 99999999999999.0
    else:
        end_time = round(trigger_time + estimated_time_sec * time_factor, 2)

    reward = format_amount(reward_val)

    if task_type == "A":
        priority = False
    elif task_type == "B":
        priority = False
    else:  # "C"
        priority = True

    type_name = TYPE_NAME_MAP[task_type]
    item_name = rng.choice(TYPE_ITEM_TITLES[task_type])
    title = f"{item_name}"
    task_id = f"{task_type}-{idx:04d}"

    return Task(
        id=task_id,
        type=task_type,
        type_name=type_name,
        title=title,
        trigger_time=trigger_time,
        pickup=pickup,
        deliver=deliver,
        distance_m=float(f"{distance_m:.2f}"),
        end_time=end_time,
        reward=reward,
        priority=priority,
    )

# 从指定区域内随机采样一个点
def sample_point(area_size_m: int, rng: random.Random) -> Point:
    return Point(x=rng.uniform(0, area_size_m), y=rng.uniform(0, area_size_m))


def generate_tasks(
    num_tasks: int,
    area_size_m: int,
    min_gap_s: int,
    max_gap_s: int,
    weights: Tuple[int, int, int],
    seed: int,
    level: int,
    speed: float
) -> List[Task]:
    rng = random.Random(seed)

    # types distribution (exact counts) and shuffle for variety
    types = generate_distribution(num_tasks, weights)
    rng.shuffle(types)

    tasks: List[Task] = []
    prev_end_time = 0.0
    
    # 如果任务level为1，则所有任务触发时间均为0
    # 否则按正常逻辑生成触发时间：下一个任务的start_time在上一个任务的end_time附近随机
    for idx in range(1, num_tasks + 1):
        if level == 1:
            trigger_time = 0.0
        else:
            if idx == 1:
                trigger_time = 0.0
            else:
                # 在上一个任务的end_time附近随机生成start_time
                # 浮动范围：end_time的-20%到+10%
                float_ratio = rng.uniform(-0.2, 0.1)
                trigger_time = round(prev_end_time * (1 + float_ratio), 2)

        pickup = sample_point(area_size_m, rng)
        deliver = sample_point(area_size_m, rng)
        task_type = types[idx - 1]

        task = build_task(
            idx=idx,
            task_type=task_type,
            trigger_time=trigger_time,
            pickup=pickup,
            deliver=deliver,
            rng=rng,
            level=level,
            speed=speed
        )
        tasks.append(task)
        
        # 更新prev_end_time为当前任务的end_time，用于下一个任务的start_time计算
        prev_end_time = task.end_time

    # ensure tasks are sorted by trigger time
    tasks.sort(key=lambda t: t.trigger_time)
    return tasks


def to_serializable(task: Task) -> Dict:
    d = asdict(task)
    # Flatten nested dataclasses for JSON readability
    d["pickup"] = asdict(task.pickup)
    d["deliver"] = asdict(task.deliver)
    return d


def write_json(output_path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="配送任务自动生成器：按A/B/C(7:2:1)规则生成任务并输出JSON"
    )
    parser.add_argument("--num_tasks", type=int, default=10, help="生成任务总数")
    parser.add_argument("--area_size", type=int, default=3000, help="坐标区域边长(米)")
    parser.add_argument("--min_gap", type=int, default=0, help="相邻任务触发时间最小间隔(秒)")
    parser.add_argument("--max_gap", type=int, default=600, help="相邻任务触发时间最大间隔(秒)")
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        choices=[1, 2],
        help="难度等级控制",
    )
    parser.add_argument(
        "--ratio",
        type=str,
        default="1,0,0",
        help="A,B,C的整数权重(例如 7,2,1)",
    )
    parser.add_argument("--seed", type=int, default=1, help="随机种子，保证可复现")
    # 兼容 agent 输入上下文的可选字段
    parser.add_argument("--speed", type=float, default=500.0, help="代理速度 (m/min)，与样例一致为500")
    parser.add_argument("--capacity", type=int, default=5, help="最大并行携带数量")
    parser.add_argument("--start_x", type=float, default=0.0, help="起始X坐标")
    parser.add_argument("--start_y", type=float, default=0.0, help="起始Y坐标")
    parser.add_argument(
        "--output",
        type=str,
        default="./test_output/a-task-l1-10-seed1.json",
        help="输出JSON文件路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        w_a, w_b, w_c = [int(x) for x in args.ratio.split(",")]
    except Exception as e:
        raise SystemExit(f"无效的ratio参数: {args.ratio}. 期望格式如 7,2,1") from e

    # 根据等级覆盖部分参数（仅当提供 --level 时生效）

    tasks = generate_tasks(
        num_tasks=args.num_tasks,
        area_size_m=args.area_size,
        min_gap_s=args.min_gap,
        max_gap_s=args.max_gap,
        weights=(w_a, w_b, w_c),
        seed=args.seed,
        level=args.level,
        speed=args.speed
    )

    # 基础元数据
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "num_tasks": args.num_tasks,
            "area_size": args.area_size,
            "min_gap": args.min_gap,
            "max_gap": args.max_gap,
            "ratio": [w_a, w_b, w_c],
            "ratio_effective": list((w_a, w_b, w_c)),
            "seed": args.seed,
        },
        "summary": {
            "count_total": len(tasks),
            "count_A": sum(1 for t in tasks if t.type == "A"),
            "count_B": sum(1 for t in tasks if t.type == "B"),
            "count_C": sum(1 for t in tasks if t.type == "C"),
        },
        "LEVEL": args.level if args.level else None,
        "SPEED": float(args.speed),
        "NOW_TIME": 0,
        "CUR_X": 0.0,
        "CUR_Y": 0.0,
        "CAPACITY": args.capacity,
        "tasks": [to_serializable(t) for t in tasks],
    }


    write_json(args.output, payload)
    print(f"已生成 {len(tasks)} 条任务 -> {args.output}")


if __name__ == "__main__":
    main()