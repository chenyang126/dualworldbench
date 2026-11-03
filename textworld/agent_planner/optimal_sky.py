import json
import os
import sys
import math
import logging
import argparse
from typing import List, Dict, Any, Tuple, Optional

# --- Configuration ---
logger = logging.getLogger("GodViewSolver")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class GodViewPlanner:
    """
    上帝视角求解器 (Offline DP Solver)
    通过状态压缩 DP 计算理论上的最大收益和最短时间。
    """
    def __init__(self, data: Dict[str, Any], speed_mps: float):
        self.tasks = sorted(data['tasks'], key=lambda x: x['trigger_time'])
        self.speed = speed_mps
        self.capacity = data.get('CAPACITY', 5)
        self.start_pos = {"x": data.get('CUR_X', 0.0), "y": data.get('CUR_Y', 0.0)}
        
        self.num_tasks = len(self.tasks)
        self.task_to_idx = {task['id']: i for i, task in enumerate(self.tasks)}

        # 构建关键点列表: [start, pickup_1...N, deliver_1...N]
        # 索引映射: 0=Start, 1~N=Pickup, N+1~2N=Deliver
        self.points = [self.start_pos]
        self.point_info = [{'type': 'start', 'trigger_time': 0.0}] 
        
        for task in self.tasks:
            self.points.append(task['pickup'])
            self.point_info.append({'type': 'pickup', 'trigger_time': task['trigger_time']})
        
        for task in self.tasks:
            self.points.append(task['deliver'])
            self.point_info.append({'type': 'deliver', 'trigger_time': task['trigger_time']})

        self.num_points = len(self.points)

    def _calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        return math.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2)

    def _get_penalty_rate(self, t_type: str) -> float:
        return {"A": 0.1, "B": 0.2, "C": 0.3}.get(t_type, 0.3)

    def _get_task_info_from_point(self, point_idx: int) -> Tuple[int, str]:
        """根据点索引返回 (任务索引, 动作类型)"""
        if 1 <= point_idx <= self.num_tasks:
            return point_idx - 1, 'pickup'
        else:
            return point_idx - (self.num_tasks + 1), 'deliver'

    def solve(self) -> Dict[str, Any]:
        """运行状态压缩 DP 寻找全局最优解"""
        logger.info(f"Starting DP Solver. Tasks: {self.num_tasks}, Capacity: {self.capacity}")
        
        # DP State: dp[delivered_mask][carried_mask][last_point_idx] = (max_reward, min_time)
        # 使用字典存储稀疏状态
        dp: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
        parent: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}

        # Initial State
        dp[(0, 0, 0)] = (0.0, 0.0)

        # Iteration
        for delivered_mask in range(1 << self.num_tasks):
            for carried_mask in range(1 << self.num_tasks):
                # Pruning: Cannot carry what is already delivered
                if (delivered_mask & carried_mask) != 0:
                    continue
                
                for last_pt_idx in range(self.num_points):
                    state = (delivered_mask, carried_mask, last_pt_idx)
                    if state not in dp:
                        continue
                    
                    curr_reward, curr_time = dp[state]

                    # Priority Logic Check (Task C Priority)
                    is_priority_mode = False
                    force_destinations = set()
                    #priority_deliver_pt_idx = -1
                    for i in range(self.num_tasks):
                        if (carried_mask >> i) & 1 and self.tasks[i]['type'] == 'C':
                            is_priority_mode = True
                            #priority_deliver_pt_idx = self.num_tasks + 1 + i
                            force_destinations.add(1 + self.num_tasks + i) 
                            #break
                    
                    # Try transition to next point
                    for next_pt_idx in range(1, self.num_points):
                        # 1. Priority Constraint
                        if is_priority_mode and next_pt_idx not in force_destinations:
                            continue

                        task_idx, action_type = self._get_task_info_from_point(next_pt_idx)

                        # 2. Validity Checks
                        if (delivered_mask >> task_idx) & 1: continue # Already delivered
                        if action_type == 'pickup':
                            if (carried_mask >> task_idx) & 1: continue # Already carried
                            if bin(carried_mask).count('1') >= self.capacity: continue # Capacity full
                        if action_type == 'deliver':
                            if not ((carried_mask >> task_idx) & 1): continue # Not carrying

                        # 3. Cost Calculation
                        dist = self._calculate_distance(self.points[last_pt_idx], self.points[next_pt_idx])
                        travel_time = dist / self.speed
                        arrival_time = curr_time + travel_time

                        # Wait for trigger time (only for pickup)
                        if action_type == 'pickup':
                            action_time = max(arrival_time, self.tasks[task_idx]['trigger_time'])
                        else:
                            action_time = arrival_time

                        # Reward Calculation
                        reward_gain = 0.0
                        new_delivered = delivered_mask
                        new_carried = carried_mask

                        if action_type == 'pickup':
                            new_carried |= (1 << task_idx)
                        else:
                            task = self.tasks[task_idx]
                            late = max(0, action_time - task['end_time'])
                            penalty = late * self._get_penalty_rate(task['type'])
                            reward_gain = max(0, task['reward'] - penalty)
                            new_delivered |= (1 << task_idx)
                            new_carried &= ~(1 << task_idx)

                        new_reward = curr_reward + reward_gain

                        # 4. State Update
                        next_state = (new_delivered, new_carried, next_pt_idx)
                        exist_reward, exist_time = dp.get(next_state, (-1.0, float('inf')))

                        # Optimize: Max Reward, then Min Time
                        if new_reward > exist_reward or (abs(new_reward - exist_reward) < 1e-6 and action_time < exist_time):
                            dp[next_state] = (new_reward, action_time)
                            parent[next_state] = state

        # Find Best Final State
        best_reward, best_time = 0.0, float('inf')
        best_final_state = None

        # We look for states where all tasks are delivered? 
        # Or simply the state with max reward (sometimes not all can be delivered profitably? 
        # Usually we want max reward regardless of completion count, but generally max reward implies completion).
        for state, (reward, time) in dp.items():
            if reward > best_reward or (abs(reward - best_reward) < 1e-6 and time < best_time):
                best_reward = reward
                best_time = time
                best_final_state = state
        
        if best_final_state is None:
            logger.warning("No valid path found.")
            return {"summary": {"revenue": 0, "time": 0}, "path": []}

        path = self._reconstruct_path(best_final_state, parent)
        return self._format_result(path, best_reward, best_time)

    def _reconstruct_path(self, final_state, parent):
        path = []
        curr = final_state
        while curr in parent:
            prev = parent[curr]
            path.append(curr)
            curr = prev
        path.reverse()
        return path

    def _format_result(self, path_states: List[tuple], total_rev: float, final_time: float):
        result_log = []
        last_time = 0.0
        last_pos = self.start_pos
        cumulative_rev = 0.0

        for _, _, pt_idx in path_states:
            task_idx, action_type = self._get_task_info_from_point(pt_idx)
            task = self.tasks[task_idx]
            curr_pos = self.points[pt_idx]
            
            dist = self._calculate_distance(last_pos, curr_pos)
            arrival_time = last_time + dist / self.speed
            
            wait_time = 0.0
            if action_type == 'pickup':
                action_time = max(arrival_time, task['trigger_time'])
                wait_time = max(0, action_time - arrival_time)
            else:
                action_time = arrival_time
            
            step_info = {
                "type": action_type,
                "id": task['id'],
                "time": round(action_time, 2),
                "pos": curr_pos,
                "wait": round(wait_time, 2)
            }

            if action_type == 'deliver':
                late = max(0, action_time - task['end_time'])
                penalty = late * self._get_penalty_rate(task['type'])
                rev = max(0, task['reward'] - penalty)
                step_info["revenue"] = round(rev, 2)
                cumulative_rev += rev

            result_log.append(step_info)
            last_time = action_time
            last_pos = curr_pos

        return {
            "summary": {
                "total_revenue": round(total_rev, 2),
                "final_time": round(final_time, 2),
                "steps": len(result_log)
            },
            "logs": result_log
        }

# ==========================================
# Main Entry
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="God View DP Solver for Delivery")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--result", default=None, help="Path to save result JSON")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file {args.input} not found.")
        sys.exit(1)

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load input: {e}")
        sys.exit(1)

    # Config Loading (Align with Simulator)
    cfg = data.get("config", {})
    speed = cfg.get("speed", data.get("SPEED", 500.0)) / 60.0 # Convert to m/s
    
    # Run Solver
    solver = GodViewPlanner(data, speed_mps=speed)
    result = solver.solve()

    # Output
    print("-" * 30)
    print(f"DP Optimization Complete.")
    print(f"Max Revenue: {result['summary']['total_revenue']:.2f}")
    print(f"Min Time:    {result['summary']['final_time']:.2f}s")
    print("-" * 30)

    # Save Results
    if args.result:
        result_dir = os.path.dirname(args.result)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        with open(args.result, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Result saved to {args.result}")
    else:
        # If no result file specified, print JSON to stdout for piping
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
