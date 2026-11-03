import json
import os
import sys
import logging
import argparse
from typing import List, Dict, Optional

# --- Imports ---
from common import BasePlanner, calculate_distance, interpolate_pos
from planner_algo import AlgoPlanner
from planner_dp import DPPlanner

logger = logging.getLogger("Simulator")

class EventDrivenSimulator:
    def __init__(self, planner: BasePlanner, speed=8.33, capacity=5, start_pos=None, result_file=None, planlog_file=None):
        self.speed = speed
        self.capacity = capacity
        self.pos = start_pos if start_pos else {"x": 0, "y": 0}
        self.now = 0.0
        self.planner = planner
        
        # State
        self.carried_tasks: List[Dict] = []
        self.completed_ids: List[str] = []
        self.interrupt_info: Optional[Dict] = None
        self.replan_reason = None
        
        # Logs
        self.execution_log: List[Dict] = []
        self.error_log: List[Dict] = []
        self.plan_history: List[Dict] = []
        
        # Control
        self.current_plan = []
        self.idle_counter = 0
        self.MAX_IDLE_STEPS = 5
        self.DIST_THRESHOLD = 5.0
        
        # Files
        self.result_file = result_file
        self.planlog_file = planlog_file

    def _get_penalty_rate(self, t_type: str) -> float:
        return {"A": 0.1, "B": 0.2, "C": 0.3}.get(t_type, 0.3)

    def _record_error(self, type_str, msg, action):
        logger.error(f"[Error] {type_str}: {msg}")
        self.error_log.append({
            "time": self.now, "type": type_str, "message": msg,
            "robot_pos": self.pos.copy(), "action": action
        })
        self.replan_reason = "error"
        # Trigger save on error for debugging
        self.save_logs()

    def run(self, all_tasks: List[Dict]):
        # 1. Preprocessing
        all_tasks = sorted(all_tasks, key=lambda x: x.get('trigger_time', 0.0))
        total_count = len(all_tasks)
        active_pool: List[Dict] = []
        next_idx = 0

        logger.info(f"Simulation Start. Total Tasks: {total_count}")

        # 2. Main Loop
        while len(self.completed_ids) < total_count:
            # --- 2.1 Update Environment (New Tasks) ---
            new_added = False
            # Check for tasks that have appeared up to current time (with slight tolerance)
            while next_idx < total_count:
                t = all_tasks[next_idx]
                if t.get('trigger_time', 0.0) <= self.now + 1e-2:
                    # Add if not completed and not currently tracked
                    if t['id'] not in self.completed_ids and \
                       not any(x['id'] == t['id'] for x in self.carried_tasks + active_pool):
                        active_pool.append(t)
                        new_added = True
                        logger.info(f"[Trigger] T={self.now:.2f}s New Task: {t['id']}")
                    next_idx += 1
                else: 
                    break

            # --- 2.2 Decision Making (Planning) ---
            # Plan if: New tasks arrived OR We have no plan but have work to do
            need_plan = new_added or (not self.current_plan and (active_pool or self.carried_tasks))
            
            if need_plan:
                # If we were interrupted previously, self.pos is already updated to the stop point.
                
                context = {
                    "speed": self.speed, 
                    "now": self.now, 
                    "pos": self.pos.copy(), 
                    "max_capacity": self.capacity, 
                    "carried": self.carried_tasks,
                    "tasks": active_pool,
                    "last_interrupt": self.interrupt_info
                }
                
                # Determine reason for logging
                if new_added: trigger_reason = "new_task"
                elif self.replan_reason: trigger_reason = self.replan_reason
                else: trigger_reason = "finished_actions"
                
                # Reset ephemeral state
                self.replan_reason = None 
                self.interrupt_info = None 
                
                logger.info(f"[Replan] T={self.now:.2f}s Pool={len(active_pool)} Reason={trigger_reason}")
                
                # *** Call Planner ***
                raw_actions = self.planner.plan(context)
                self.current_plan = raw_actions if raw_actions else []

                self.plan_history.append({
                    "time": self.now, 
                    "trigger": trigger_reason,
                    "pos": self.pos.copy(), 
                    "plan": self.current_plan.copy()
                })

                # Deadlock Prevention
                if not self.current_plan:
                    self.idle_counter += 1
                    if self.idle_counter >= self.MAX_IDLE_STEPS:
                        if next_idx >= total_count:
                            logger.error("FATAL: Deadlock with no future tasks.")
                            break
                        # Jump to next event
                        next_t = all_tasks[next_idx]['trigger_time']
                        logger.warning(f"Deadlock -> Jumping time to {next_t:.2f}s")
                        self.now = next_t
                        self.idle_counter = 0
                    else:
                        self.now += 1.0
                    continue
                self.idle_counter = 0

            # --- 2.3 Execution ---
            interrupted = False
            next_trigger_time = all_tasks[next_idx]['trigger_time'] if next_idx < total_count else float('inf')

            while self.current_plan and not interrupted:
                action = self.current_plan[0] # Peek first
                act_type = action.get("action")

                if act_type == "move":
                    target = action.get("to")
                    if not target: 
                        self.current_plan.pop(0)
                        continue
                        
                    dist = calculate_distance(self.pos, target)
                    duration = dist / self.speed
                    
                    # Check for Interruption
                    if self.now + duration > next_trigger_time:
                        ratio = (next_trigger_time - self.now) / duration
                        
                        # [新增 1] 记录出发点（可选，方便调试）
                        start_pos_debug = self.pos.copy()
                        
                        # 物理更新：计算中断点的坐标
                        self.pos = interpolate_pos(self.pos, target, ratio)
                        self.now = next_trigger_time
                        
                        # [新增 2] 关键补丁：将这次“未完成的移动”写入日志
                        # 这样你的日志中就会出现一段从起点到中断点的 move 记录
                        self.execution_log.append({
                            "type": "move",
                            "to": self.pos.copy(),      # 实际停下的位置
                            "time": self.now,
                            "note": "interrupted"       # 标记：这是被打断的移动
                        })
                        
                        self.interrupt_info = {
                            "stopped_at": self.pos.copy(),
                            "original_target": target,
                            "reason": "New Task Arrival"
                        }
                        self.replan_reason = "interrupt"
                        interrupted = True # Breaks the inner loop
                        self.current_plan = [] # Force replan
                        logger.info(f"[Interrupt] Moving interrupted at {self.now:.2f}s.")
                    else:
                        # Complete Move
                        self.pos = target
                        self.now += duration
                        self.execution_log.append({"type": "move", "to": target, "time": self.now})
                        self.current_plan.pop(0)

                elif act_type in ["pickup", "deliver"]:
                    tid = action.get("task_id")
                    pool = active_pool if act_type == "pickup" else self.carried_tasks
                    obj = next((t for t in pool if t['id'] == tid), None)

                    if not obj:
                        self._record_error("TASK_NOT_FOUND", f"Task {tid} missing", action)
                        interrupted = True
                        self.current_plan = []
                        break

                    target_pt = obj['pickup'] if act_type == "pickup" else obj['deliver']
                    
                    # Geometry Check
                    dist_to_target = calculate_distance(self.pos, target_pt)
                    if dist_to_target > self.DIST_THRESHOLD:
                        self._record_error("INVALID_LOCATION", 
                                           f"Too far from target ({dist_to_target:.2f}m)", action)
                        interrupted = True
                        self.current_plan = []
                        break

                    if act_type == "pickup":
                        if len(self.carried_tasks) >= self.capacity:
                            self._record_error("OVERLOAD", "Capacity full", action)
                            interrupted = True
                            self.current_plan = []
                            break
                        self.carried_tasks.append(obj)
                        active_pool = [t for t in active_pool if t['id'] != tid]
                        self.execution_log.append({"type": "pickup", "id": tid, "time": self.now})
                    else:
                        self.carried_tasks = [t for t in self.carried_tasks if t['id'] != tid]
                        self.completed_ids.append(tid)
                        late = max(0, self.now - obj['end_time'])
                        rev = max(0, obj['reward'] - late * self._get_penalty_rate(obj['type']))
                        self.execution_log.append({"type": "deliver", "id": tid, "time": self.now, "revenue": rev})
                        logger.info(f"[Deliver] {tid} Rev={rev:.2f}")

                    self.current_plan.pop(0)

                else:
                    # Unknown action, skip
                    self.current_plan.pop(0)

            # --- 2.4 Idle Time Jump ---
            # If we are idle and waiting for the next task
            if not interrupted and not self.current_plan and next_idx < total_count:
                if self.now < next_trigger_time:
                    self.now = next_trigger_time

    def save_logs(self):
        """Save results to file. Separated to avoid I/O blocking in simulation loop."""
        total_rev = sum(l.get('revenue', 0) for l in self.execution_log if l['type'] == 'deliver')
        res = {
            "summary": {
                "total_revenue": round(total_rev, 2),
                "final_time": round(self.now, 2),
                "errors": len(self.error_log)
            },
            "logs": self.execution_log,
            "errors": self.error_log
        }
        
        if self.result_file:
            result_dir = os.path.dirname(self.result_file)
            if result_dir:  # Only create directory if path contains a directory
                os.makedirs(result_dir, exist_ok=True)
            with open(self.result_file, 'w', encoding='utf-8') as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
        
        if self.planlog_file:
            planlog_dir = os.path.dirname(self.planlog_file)
            if planlog_dir:  # Only create directory if path contains a directory
                os.makedirs(planlog_dir, exist_ok=True)
            with open(self.planlog_file, 'w', encoding='utf-8') as f:
                json.dump(self.plan_history, f, indent=2, ensure_ascii=False)
        return res

# ==========================================
# Main Entry
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Universal Delivery Simulator")
    parser.add_argument("--mode", choices=["llm", "algo", "dp"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--planlog", required=True)
    parser.add_argument("--model", default="gpt-5")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Config loading
    cfg = data.get("config", {})
    # Fallback logic simplified
    speed = cfg.get("speed", data.get("SPEED", 500.0)) / 60.0
    capacity = cfg.get("capacity", data.get("CAPACITY", 5))
    start_pos = {
        "x": cfg.get("start_x", data.get("CUR_X", 0.0)), 
        "y": cfg.get("start_y", data.get("CUR_Y", 0.0))
    }

    if args.mode == "llm":
        print(f"Init GPTPlanner ({args.model})")
        from planner_gpt import GPTPlanner
        planner = GPTPlanner(model=args.model)
    elif args.mode == "dp":
        print(f"Init DPPlanner")
        planner = DPPlanner()
    else:
        print("Init AlgoPlanner (DFS)")
        planner = AlgoPlanner()

    sim = EventDrivenSimulator(planner, speed=speed, capacity=capacity, start_pos=start_pos, 
                               result_file=args.result, planlog_file=args.planlog)
    sim.run(data['tasks'])
    
    # Final save
    res = sim.save_logs()

    print("-" * 30)
    print(f"Simulation Complete. Rev: {res['summary']['total_revenue']:.2f}")
    print(f"Saved to: {args.result}")

if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()