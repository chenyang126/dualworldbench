import time
import math
from typing import Any, Dict, List, Set, Tuple
from common import BasePlanner, calculate_distance

class AlgoSolver:
    """
    DFS Solver with Branch & Bound and Greedy Warm-up.
    Optimized for Type-C priority constraints.
    """
    def __init__(self, context: Dict[str, Any], time_limit=2.0):
        self.start_time = context['now']
        self.start_pos = context['pos']
        self.start_load = len(context['carried'])
        self.capacity = context['max_capacity']
        self.speed = context['speed']
        self.time_limit = time_limit
        self.start_clock = time.time()
        
        # Merge all tasks into a single list for indexing
        # Store as dicts, add 'state' implicitly via sets
        self.tasks = context['carried'] + context['tasks']
        self.num_tasks = len(self.tasks)
        
        # Indices management
        self.initial_carrying = set(range(len(context['carried'])))
        self.initial_to_pickup = set(range(len(context['carried']), self.num_tasks))
        
        # Pre-calculate constants to avoid repeated dict lookups
        self.PENALTY_RATES = {'A': 0.1, 'B': 0.2, 'C': 0.3}
        
        # Global best tracking
        self.best_revenue = -float('inf')
        self.best_path = []

    def solve(self):
        if not self.tasks:
            return []

        # 1. Greedy Warm-up
        # Run a quick greedy pass to establish a baseline high score.
        # This makes the pruning in DFS effective immediately.
        self._greedy_search()

        # 2. DFS (Branch & Bound)
        # Only run if we have time left
        if time.time() - self.start_clock < self.time_limit:
            self._dfs(
                curr_pos=self.start_pos,
                curr_time=self.start_time,
                curr_load=self.start_load,
                curr_rev=0.0,
                carrying=self.initial_carrying,
                pending_pickup=self.initial_to_pickup,
                path=[]
            )

        return self._format_output()

    def _greedy_search(self):
        """
        Non-recursive greedy approach to find a 'good enough' solution fast.
        """
        curr_pos = self.start_pos
        curr_time = self.start_time
        curr_load = self.start_load
        curr_rev = 0.0
        
        carrying = self.initial_carrying.copy()
        pending = self.initial_to_pickup.copy()
        path = []

        while carrying or pending:
            # Generate valid actions
            actions = self._get_valid_actions(curr_pos, curr_time, curr_load, carrying, pending)
            if not actions: break # Should not happen unless deadlock
            
            # Pick the one with the best immediate score (Greedy)
            # Sort by score (asc) -> pick first
            actions.sort(key=lambda x: x['score'])
            best_act = actions[0]
            
            # Update State
            path.append(best_act)
            curr_pos = best_act['pos']
            curr_time = best_act['finish_time']
            curr_rev += best_act['revenue']
            
            idx = best_act['idx']  # Use index, not task_id
            if best_act['type'] == 'pickup':
                pending.remove(idx)
                carrying.add(idx)
                curr_load += 1
            else:
                carrying.remove(idx)
                curr_load -= 1
        
        # Save this as the baseline
        if curr_rev > self.best_revenue:
            self.best_revenue = curr_rev
            self.best_path = path

    def _dfs(self, curr_pos, curr_time, curr_load, curr_rev, carrying: Set[int], pending_pickup: Set[int], path: List):
        # 1. Timeout Check (Check every 10 steps or just rely on overhead)
        if (len(path) % 5 == 0) and (time.time() - self.start_clock > self.time_limit):
            return

        # 2. Termination
        if not carrying and not pending_pickup:
            if curr_rev > self.best_revenue:
                self.best_revenue = curr_rev
                self.best_path = list(path)
            return

        # 3. Pruning (Branch & Bound)
        potential = self._estimate_potential(curr_pos, curr_time, carrying, pending_pickup)
        if curr_rev + potential <= self.best_revenue:
            return

        # 4. Expansion
        candidates = self._get_valid_actions(curr_pos, curr_time, curr_load, carrying, pending_pickup)
        
        # Sort candidates to find better solutions earlier (Heuristic)
        candidates.sort(key=lambda x: x['score'])

        for act in candidates:
            idx = act['idx']  # Use index, not task_id
            
            # Create new state sets (Copy overhead is acceptable for N<20)
            new_carrying = carrying.copy()
            new_pending = pending_pickup.copy()
            
            new_load = curr_load
            if act['type'] == 'pickup':
                new_pending.remove(idx)
                new_carrying.add(idx)
                new_load += 1
            else:
                new_carrying.remove(idx)
                new_load -= 1
            
            self._dfs(
                act['pos'], 
                act['finish_time'], 
                new_load, 
                curr_rev + act['revenue'], 
                new_carrying, 
                new_pending, 
                path + [act]
            )

    def _get_valid_actions(self, curr_pos, curr_time, curr_load, carrying: Set[int], pending: Set[int]) -> List[Dict]:
        actions = []
        
        # --- Constraint Check: Type C Priority ---
        # If we are carrying any Type C task, we CANNOT deliver A or B.
        carrying_c_ids = [i for i in carrying if self.tasks[i]['type'] == 'C']
        is_carrying_c = len(carrying_c_ids) > 0

        # 1. Pickup Options
        if curr_load < self.capacity:
            for idx in pending:
                task = self.tasks[idx]
                dist = calculate_distance(curr_pos, task['pickup'])
                arr_time = curr_time + dist / self.speed
                # Wait if arrived early
                start_act = max(arr_time, task.get('trigger_time', 0))
                
                # Heuristic Score
                score = start_act
                if task['type'] == 'C': score -= 2000 # Prioritize picking up C

                actions.append({
                    "type": "pickup", "task_id": task['id'], "idx": idx, # internal index
                    "pos": task['pickup'], "finish_time": start_act,
                    "revenue": 0, "score": score
                })

        # 2. Delivery Options
        for idx in carrying:
            task = self.tasks[idx]
            
            # Constraint: If carrying C, can only deliver C
            if is_carrying_c and task['type'] != 'C':
                continue

            dist = calculate_distance(curr_pos, task['deliver'])
            arr_time = curr_time + dist / self.speed
            
            # Calculate Revenue
            rate = self.PENALTY_RATES.get(task['type'], 0.3)
            late = max(0, arr_time - task.get('end_time', 0))
            rev = max(0, task.get('reward', 0) - late * rate)
            
            # Heuristic Score: Earlier finish - (Revenue * Weight)
            score = arr_time - (rev * 10)
            if task['type'] == 'C': score -= 5000 # Must deliver C ASAP

            actions.append({
                "type": "deliver", "task_id": task['id'], "idx": idx,
                "pos": task['deliver'], "finish_time": arr_time,
                "revenue": rev, "score": score
            })
            
        return actions

    def _estimate_potential(self, curr_pos, curr_time, carrying, pending):
        """
        Upper Bound Estimation.
        Calculates the maximum possible revenue remaining.
        Includes minimal travel time penalty to make the bound tighter.
        """
        potential = 0.0
        
        # Merge indices to check
        all_indices = list(carrying) + list(pending)
        
        for idx in all_indices:
            task = self.tasks[idx]
            reward = task.get('reward', 0)
            rate = self.PENALTY_RATES.get(task['type'], 0.3)
            end_time = task.get('end_time', 0)
            
            # Minimal geometry bound:
            # We must at least travel from current -> Deliver (if carrying)
            # or current -> Pickup -> Deliver (if pending)
            # This is a weak bound but better than 0 latency.
            if idx in carrying:
                dist = calculate_distance(curr_pos, task['deliver'])
            else:
                # Approximate: dist to pickup + pickup to deliver
                # We ignore curr->pickup to be safe (optimistic), 
                # but we MUST travel pickup->deliver.
                d1 = calculate_distance(task['pickup'], task['deliver'])
                dist = d1 
            
            min_duration = dist / self.speed
            min_arrival = curr_time + min_duration
            
            late = max(0, min_arrival - end_time)
            max_rev = max(0, reward - late * rate)
            potential += max_rev
            
        return potential

    def _format_output(self):
        actions_list = []
        current_pos = self.start_pos
        
        for step in self.best_path:
            target_pos = step["pos"]
            dist = calculate_distance(current_pos, target_pos)
            
            # Insert Move action if physically apart
            if dist > 0.01:
                actions_list.append({"action": "move", "to": target_pos})
                
            actions_list.append({"action": step["type"], "task_id": step["task_id"]})
            current_pos = target_pos
            
        return actions_list

class AlgoPlanner(BasePlanner):
    def plan(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        solver = AlgoSolver(context, time_limit=2.0)
        return solver.solve()