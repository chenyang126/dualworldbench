import logging
from typing import List, Dict, Tuple, Any
from common import BasePlanner, calculate_distance

logger = logging.getLogger("DPPlanner")

class DPPlanner(BasePlanner):
    def __init__(self):
        super().__init__()
        self.penalty_map = {"A": 0.1, "B": 0.2, "C": 0.3, "default": 0.3}

    def _get_penalty(self, t_type: str) -> float:
        return self.penalty_map.get(t_type, self.penalty_map["default"])

    def plan(self, context: Dict) -> List[Dict]:
        """
        使用状态压缩动态规划来制定最优的行动序列。
        """
        # 1. 解包上下文信息
        self.speed = context['speed']
        start_time = context['now']
        start_pos = context['pos']
        self.capacity = context['max_capacity']
        carried_tasks_list = context['carried']
        pool_tasks_list = context['tasks']
        # 如果没有任何待办事项，直接返回空计划
        if not carried_tasks_list and not pool_tasks_list:
            return []

        # 2. 数据预处理：构建DP所需的标准数据结构
        # 将所有任务统一编号，并创建位置点列表
        tasks = carried_tasks_list + pool_tasks_list
        num_tasks = len(tasks)
        task_map = {task['id']: task for task in tasks}
        task_to_idx = {task['id']: i for i, task in enumerate(tasks)}

        # points 列表: [start_pos, p0, p1, ..., d0, d1, ...]
        points = [start_pos]
        # pickup_start_idx=1, deliver_start_idx=1+num_tasks
        for task in tasks:
            points.append(task['pickup'])
        for task in tasks:
            points.append(task['deliver'])
        
        num_points = len(points)

        # 3. 初始化DP表和父节点追溯表
        # DP状态: dp[delivered_mask][carried_mask][last_point_idx] = (max_reward, min_time)
        # delivered_mask: 已交付任务的位掩码
        # carried_mask:   当前携带任务的位掩码
        # last_point_idx: 当前所在位置点的索引
        dp = {}  # 使用字典来节省空间，只存储可达状态
        parent = {} # 用于回溯最优路径

        # 设置初始状态
        initial_carried_mask = 0
        initial_load = 0
        for task in carried_tasks_list:
            task_idx = task_to_idx[task['id']]
            initial_carried_mask |= (1 << task_idx)
            initial_load += 1
        
        # 初始状态：位于起点(0)，携带任务由initial_carried_mask定义，已交付任务为0
        initial_state = (0, initial_carried_mask, 0)
        dp[initial_state] = (0.0, 0.0) # (reward, time)

        # 4. 运行DP状态转移
        # 我们按状态的复杂度（已交付任务数）顺序进行迭代
        for delivered_mask in range(1 << num_tasks):
            for carried_mask in range(1 << num_tasks):
                # 剪枝：已交付的任务必须是当前携带或曾经携带过的
                if (delivered_mask & carried_mask) != 0: continue
                
                for last_pt_idx in range(num_points):
                    state = (delivered_mask, carried_mask, last_pt_idx)
                    if state not in dp:
                        continue
                    
                    current_reward, current_time = dp[state]
                    
                    # 检查是否处于“优先交付C任务”模式
                    priority_mode = False
                    #priority_deliver_pt_idx = -1
                    force_destinations = set()
                    current_load = bin(carried_mask).count('1')

                    for i in range(num_tasks):
                        if (carried_mask >> i) & 1: # 如果携带任务i
                            if tasks[i]['type'] == 'C':
                                priority_mode = True
                                force_destinations.add(1 + num_tasks + i) 
                                #priority_deliver_pt_idx = 1 + num_tasks + i
                                #break
                    
                    # 尝试转移到下一个状态
                    for next_pt_idx in range(1, num_points):
                        # A. 合法性检查
                        # 规则1: 处于优先模式时，只能去C的交付点
                        if priority_mode and next_pt_idx not in force_destinations:
                            continue
                        
                        task_idx = -1
                        action_type = ''
                        if 1 <= next_pt_idx < 1 + num_tasks:
                            task_idx = next_pt_idx - 1
                            action_type = 'pickup'
                        else:
                            task_idx = next_pt_idx - (1 + num_tasks)
                            action_type = 'deliver'

                        # 规则2: 不能访问已交付任务的任何点
                        if (delivered_mask >> task_idx) & 1:
                            continue
                        
                        # 规则3: 不能重复拾取
                        if action_type == 'pickup' and (carried_mask >> task_idx) & 1:
                            continue

                        # 规则4: 必须携带了才能交付
                        if action_type == 'deliver' and not ((carried_mask >> task_idx) & 1):
                            continue
                        
                        # 规则5: 不能超载
                        if action_type == 'pickup' and current_load >= self.capacity:
                            continue

                        # B. 计算新状态和收益
                        dist = calculate_distance(points[last_pt_idx], points[next_pt_idx])
                        travel_time = dist / self.speed
                        new_time = current_time + travel_time
                        
                        reward_gain = 0.0
                        new_delivered_mask = delivered_mask
                        new_carried_mask = carried_mask

                        if action_type == 'pickup':
                            new_carried_mask |= (1 << task_idx)
                        else: # deliver
                            arrival_at_deliver = start_time + new_time
                            task = tasks[task_idx]
                            late = max(0, arrival_at_deliver - task['end_time'])
                            penalty = late * self._get_penalty(task['type'])
                            reward_gain = max(0, task['reward'] - penalty)
                            
                            new_delivered_mask |= (1 << task_idx)
                            new_carried_mask &= ~(1 << task_idx)

                        new_reward = current_reward + reward_gain
                        
                        # C. 更新DP表
                        next_state = (new_delivered_mask, new_carried_mask, next_pt_idx)
                        
                        # 如果新路径更优，则更新
                        # 目标：1. 奖励更高 2. 奖励相同但时间更短
                        existing_reward, existing_time = dp.get(next_state, (-1.0, float('inf')))
                        if new_reward > existing_reward or \
                           (abs(new_reward - existing_reward) < 1e-6 and new_time < existing_time):
                            dp[next_state] = (new_reward, new_time)
                            parent[next_state] = state

        # 5. 查找最优解并回溯路径
        best_reward, best_time = -1.0, float('inf')
        best_final_state = None

        if not dp: return [] # 如果没有可达状态

        for state, (reward, time) in dp.items():
            if reward > best_reward or (abs(reward - best_reward) < 1e-6 and time < best_time):
                best_reward = reward
                best_time = time
                best_final_state = state
        
        if best_final_state is None:
            return []

        # 回溯
        path = []
        curr_state = best_final_state
        while curr_state in parent:
            prev_state = parent[curr_state]
            
            last_pt_idx = curr_state[2]
            
            task_idx = -1
            action_type = ''
            if 1 <= last_pt_idx < 1 + num_tasks:
                task_idx = last_pt_idx - 1
                action_type = 'pickup'
            else:
                task_idx = last_pt_idx - (1 + num_tasks)
                action_type = 'deliver'

            path.append({
                "task_id": tasks[task_idx]['id'],
                "action_type": action_type
            })
            curr_state = prev_state
        path.reverse()

        final_plan = []
        for step in path:
            task_id = step["task_id"]
            action_type = step["action_type"]
            task_info = task_map[task_id]

            target_pos = None
            if action_type == 'pickup':
                target_pos = task_info['pickup']
            elif action_type == 'deliver':
                target_pos = task_info['deliver']
            
            if target_pos:
                # 添加移动指令
                final_plan.append({
                    "action": "move",
                    "to": target_pos
                })
                # 添加具体动作指令
                final_plan.append({
                    "action": action_type,
                    "task_id": task_id
                })
        
        return final_plan

    