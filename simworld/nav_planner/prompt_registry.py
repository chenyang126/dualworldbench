from __future__ import annotations

from textwrap import dedent


PROMPT_SCENE_ALIASES = {
    "winding_road": "indoor",
    "rooms": "indoor",
    "rooms_mixed": "indoor",
    "branch_corridor": "indoor",
    "l1_spine_rooms": "simworld",
    "l1_cross_suite": "simworld",
    "l1_offset_wing": "simworld",
    "l1_dual_room": "simworld",
    "l1_corner_hall": "simworld",
    "l2_loop_rooms": "simworld",
    "l2_dual_spine": "simworld",
    "l2_ring_hub": "simworld",
    "l2_offset_blocks": "simworld",
    "l2_cluster_corridor": "simworld",
}


def _clean_prompt(text: str) -> str:
    return dedent(text).strip() + "\n"


SCENE_PROMPT_OVERRIDES: dict[str, dict[str, str]] = {
    "simworld": {
        "meta": _clean_prompt(
            """
            You are the route planner for a delivery robot in a structured indoor SimWorld benchmark scene.
            Generate one strictly sequential visiting order over the task nodes of the current task set.

            ### 1. PRIMARY OBJECTIVE
            - Minimize total topological travel cost while completing as many tasks as possible.
            - If rewards are present and differ, prefer higher total reward only when legality and travel efficiency are preserved.
            - A fully legal plan is better than a shorter but invalid one.
            - Travel cost must be judged from reachable indoor topology in the BEV map, not from straight-line distance.

            ### 2. CURRENT STATE
            - Time: {now:.2f} s
            - Position: {{ "x": {cur_x:.2f}, "y": {cur_y:.2f} }}
            - Speed: {speed:.2f} m/s
            - Current load: {capacity}/{max_capacity}

            ### 3. SIMWORLD MAP INTERPRETATION
            - The image is a BEV map of a structured indoor scene.
            - START is the green marker.
            - Pickup nodes are red markers labeled P1, P2, P3, ...
            - Delivery nodes are blue markers labeled D1, D2, D3, ...
            - Traversable space is the connected indoor free-space topology formed by rooms, openings, corridors, and structured open areas.
            - Black regions outside that connected free space are not shortcuts and must not be crossed.
            - Walls, blocked regions, and obstacle bodies cannot be crossed.
            - Two points that look close in Euclidean distance may still be expensive if a wall, partition, or detour separates them.
            - Prefer globally coherent routes that reduce backtracking across branches, room transitions, and bottlenecks.

            ### 4. TASK TO NODE MAPPING
            - Task A-0001 corresponds to pickup node P1 and delivery node D1.
            - Your answer must name these nodes only. The mapping table below tells you where each node is.
            --- MAPPING TABLE START ---
            {id_mapping_section}
            --- MAPPING TABLE END ---

            ### 5. TASK POOL
            Carried tasks:
            {carried_tasks_json}

            Available tasks:
            {new_tasks_json}

            ### 6. HARD CONSTRAINTS
            1. Pn must appear before the matching Dn.
            2. Never place Dn before the matching Pn.
            3. Each pickup node Pn may appear at most once, and each delivery node Dn may appear at most once.
            4. A task that is already carried has no Pn node; only its Dn may be visited.
            5. The number of picked but not yet delivered tasks must never exceed {max_capacity}.
            6. After every Pn, mentally increase load by 1; after every Dn, mentally decrease load by 1.
            7. Use only node labels that exist in the mapping table above. Inventing a node is a planning error.
            8. Avoid plans that are short in Euclidean distance but obviously poor in BEV topology.

            ### 7. PLANNING STRATEGY
            - Favor short topological loops or local room batches only when they remain valid under the load limit.
            - Prefer finishing nearby delivery obligations before creating excessive pending load.
            - When multiple branches exist, reduce repeated cross-branch switching unless it clearly saves downstream cost.
            - Use the BEV structure to reason about detours, not just marker proximity.

            ### 8. SELF-CHECK BEFORE YOU ANSWER
            Verify all of the following before returning JSON:
            - Every pickup node appears at most once.
            - Every delivery node appears at most once.
            - Every Dn appears after its matching Pn.
            - Load never exceeds {max_capacity} at any step.
            - Every node label appears in the mapping table.
            - The final answer is a single valid JSON object with a "sequence" array.

            ### 9. OUTPUT
            Return ONLY one raw JSON object in this format:
            {{
              "sequence": ["START", "P1", "P3", "D3", "D1"]
            }}
            The robot is assumed to travel to each node in the order you give; do not output
            coordinates, moves, or any other field.
            Do not include explanations or markdown.
            """
        ),
        "text": _clean_prompt(
            """
            You are the route planner for a delivery robot in a structured indoor SimWorld benchmark scene.
            You only receive structured task text, not the map image.

            ### 1. PRIMARY OBJECTIVE
            - Minimize estimated travel cost while completing as many tasks as possible.
            - If rewards are present and differ, prefer higher total reward only when legality is preserved.
            - A fully legal plan is better than a shorter but invalid one.

            ### 2. CURRENT STATE
            - Time: {now:.2f} s
            - Position: {{ "x": {cur_x:.2f}, "y": {cur_y:.2f} }}
            - Speed: {speed:.2f} m/s
            - Current load: {capacity}/{max_capacity}

            ### 3. AVAILABLE INFORMATION
            - You do NOT have access to the BEV map image.
            - Use the provided coordinates only as a cautious proxy for travel cost.
            - In structured indoor scenes, nearby coordinates may still be separated by walls or topology detours.
            - Prefer conservative, legality-first batching and avoid aggressive long jumps based only on Euclidean closeness.

            ### 4. TASK TO NODE MAPPING
            - Task A-0001 corresponds to pickup node P1 and delivery node D1.
            - Your answer must name these nodes only. The mapping table below gives their coordinates.
            --- MAPPING TABLE START ---
            {id_mapping_section}
            --- MAPPING TABLE END ---

            ### 5. TASK POOL
            Carried tasks:
            {carried_tasks_json}

            Available tasks:
            {new_tasks_json}

            ### 6. HARD CONSTRAINTS
            1. Pn must appear before the matching Dn.
            2. Never place Dn before the matching Pn.
            3. Each pickup node Pn may appear at most once, and each delivery node Dn may appear at most once.
            4. A task that is already carried has no Pn node; only its Dn may be visited.
            5. The number of picked but not yet delivered tasks must never exceed {max_capacity}.
            6. After every Pn, mentally increase load by 1; after every Dn, mentally decrease load by 1.
            7. Use only node labels that exist in the mapping table above. Inventing a node is a planning error.

            ### 7. SELF-CHECK BEFORE YOU ANSWER
            Verify all of the following before returning JSON:
            - Every pickup node appears at most once.
            - Every delivery node appears at most once.
            - Every Dn appears after its matching Pn.
            - Load never exceeds {max_capacity} at any step.
            - Every node label appears in the mapping table.
            - The final answer is a single valid JSON object with a "sequence" array.

            ### 8. OUTPUT
            Return ONLY one raw JSON object in this format:
            {{
              "sequence": ["START", "P1", "P3", "D3", "D1"]
            }}
            The robot is assumed to travel to each node in the order you give; do not output
            coordinates, moves, or any other field.
            Do not include explanations or markdown.
            """
        ),
        "image": _clean_prompt(
            """
            You are the route planner for a delivery robot in a structured indoor SimWorld benchmark scene.
            You only receive the BEV map image and must output a topologically valid node sequence.

            ### 1. PRIMARY OBJECTIVE
            - Minimize total topological travel cost while keeping the route valid.
            - Prefer globally coherent traversal over visually short but topologically poor hopping.

            ### 2. MAP INTERPRETATION
            - START is the green marker.
            - Pickup nodes are red markers labeled P1, P2, P3, ...
            - Delivery nodes are blue markers labeled D1, D2, D3, ...
            - Traversable space is the connected indoor free-space topology visible in the BEV map.
            - Walls, obstacle blocks, and black non-traversable regions cannot be crossed.
            - Euclidean closeness is not enough; route quality depends on openings, room transitions, branches, and detours.

            ### 3. TASK LOGIC
            - Pn must appear before the matching Dn.
            - Never place Dn before the matching Pn.
            - Each pickup node Pn may appear at most once, and each delivery node Dn may appear at most once.
            - The number of picked but not yet delivered tasks must never exceed the load capacity limit.
            - Use a conservative legality-first strategy: avoid collecting many pickups if that increases backtracking or risks capacity overflow.
            - Prefer route segments that respect visible connectivity and reduce repeated cross-branch switching.

            ### 4. SELF-CHECK BEFORE YOU ANSWER
            Verify all of the following before returning JSON:
            - Every Dn has an earlier matching Pn.
            - No pickup or delivery node is repeated.
            - The number of undelivered pickups never exceeds the allowed capacity limit.
            - The sequence is consistent with visible indoor topology.
            - The answer is a single valid JSON object with a "sequence" array.

            ### 5. OUTPUT
            Return ONLY one raw JSON object in this format:
            {{
              "sequence": ["START", "P1", "P3", "D3", "D1"]
            }}
            The robot is assumed to travel to each node in the order you give; do not output
            coordinates, moves, or any other field.
            Do not include explanations or markdown.
            """
        ),
    },
    "indoor": {
        "meta": _clean_prompt(
            """
            You are the route planner for an indoor delivery robot.
            Generate one strictly sequential visiting order over the task nodes of the current indoor task set.

            ### 1. PRIMARY OBJECTIVE
            - Maximize total collected reward while minimizing completion time.
            - If rewards are missing or equal, minimize total completion time for all feasible tasks.
            - A plan with any illegal pickup/delivery or load overflow is worse than a slightly longer but fully valid plan.
            - Travel cost must be estimated from indoor corridor / room connectivity, not from straight-line distance.

            ### 2. CURRENT STATE
            - Time: {now:.2f} s
            - Position: {{ "x": {cur_x:.2f}, "y": {cur_y:.2f} }}
            - Speed: {speed:.2f} m/s
            - Current load: {capacity}/{max_capacity}

            ### 3. INDOOR MAP INTERPRETATION
            - The map image is a BEV indoor layout.
            - START is the green marker.
            - Pickup nodes are red markers labeled P1, P2, P3, ...
            - Delivery nodes are blue markers labeled D1, D2, D3, ...
            - Traversable space is the connected corridor / room network shown in the image.
            - Black regions outside that network are not shortcuts and must not be crossed.
            - Walls and blocked regions cannot be crossed.
            - Judge travel cost by reachable indoor path length through doors, corridors, and room openings.

            ### 4. TASK TO NODE MAPPING
            - Task A-0001 corresponds to pickup node P1 and delivery node D1.
            - Your answer must name these nodes only. The mapping table below tells you where each node is.
            --- MAPPING TABLE START ---
            {id_mapping_section}
            --- MAPPING TABLE END ---

            ### 5. TASK POOL
            Carried tasks:
            {carried_tasks_json}

            Available tasks:
            {new_tasks_json}

            ### 6. HARD CONSTRAINTS
            1. Pn must appear before the matching Dn.
            2. Never place Dn before the matching Pn.
            3. Each pickup node Pn may appear at most once, and each delivery node Dn may appear at most once.
            4. A task that is already carried has no Pn node; only its Dn may be visited.
            5. The number of picked but not yet delivered tasks must never exceed {max_capacity}.
            6. After every Pn, mentally increase load by 1; after every Dn, mentally decrease load by 1.
            7. Use only node labels that exist in the mapping table above. Inventing a node is a planning error.
            8. Avoid plans that look short in Euclidean distance but are long in indoor topology.

            ### 7. PLANNING STRATEGY
            - Prefer short local batches only when they remain valid under the load limit.
            - In ambiguous cases, choose the safer legal ordering over a risky aggressive batching strategy.
            - Reduce backtracking, but never trade legality for shorter distance.

            ### 8. SELF-CHECK BEFORE YOU ANSWER
            Verify all of the following before returning JSON:
            - Every pickup node appears at most once.
            - Every delivery node appears at most once.
            - Every Dn appears after its matching Pn.
            - Load never exceeds {max_capacity} at any step.
            - Every node label appears in the mapping table.
            - The final answer is a single valid JSON object with a "sequence" array.

            ### 9. OUTPUT
            Return ONLY one raw JSON object in this format:
            {{
              "sequence": ["START", "P1", "P3", "D3", "D1"]
            }}
            The robot is assumed to travel to each node in the order you give; do not output
            coordinates, moves, or any other field.
            Do not include explanations or markdown.
            """
        ),
        "text": _clean_prompt(
            """
            You are the route planner for an indoor delivery robot.
            You only receive structured task text, not the map image.

            ### 1. PRIMARY OBJECTIVE
            - Maximize total collected reward while minimizing estimated completion time.
            - If rewards are missing or equal, minimize estimated completion time for all feasible tasks.
            - A fully legal plan is better than a shorter but invalid plan.

            ### 2. CURRENT STATE
            - Time: {now:.2f} s
            - Position: {{ "x": {cur_x:.2f}, "y": {cur_y:.2f} }}
            - Speed: {speed:.2f} m/s
            - Current load: {capacity}/{max_capacity}

            ### 3. AVAILABLE INFORMATION
            - You do NOT have access to the indoor map image.
            - Use the provided coordinates as a cautious proxy for travel cost.
            - Indoor routes are not straight lines, so avoid aggressive long jumps and over-batching.
            - Prefer conservative local batching that respects the load limit.

            ### 4. TASK TO NODE MAPPING
            - Task A-0001 corresponds to pickup node P1 and delivery node D1.
            - Your answer must name these nodes only. The mapping table below gives their coordinates.
            --- MAPPING TABLE START ---
            {id_mapping_section}
            --- MAPPING TABLE END ---

            ### 5. TASK POOL
            Carried tasks:
            {carried_tasks_json}

            Available tasks:
            {new_tasks_json}

            ### 6. HARD CONSTRAINTS
            1. Pn must appear before the matching Dn.
            2. Never place Dn before the matching Pn.
            3. Each pickup node Pn may appear at most once, and each delivery node Dn may appear at most once.
            4. A task that is already carried has no Pn node; only its Dn may be visited.
            5. The number of picked but not yet delivered tasks must never exceed {max_capacity}.
            6. After every Pn, mentally increase load by 1; after every Dn, mentally decrease load by 1.
            7. Use only node labels that exist in the mapping table above. Inventing a node is a planning error.

            ### 7. SELF-CHECK BEFORE YOU ANSWER
            Verify all of the following before returning JSON:
            - Every pickup node appears at most once.
            - Every delivery node appears at most once.
            - Every Dn appears after its matching Pn.
            - Load never exceeds {max_capacity} at any step.
            - Every node label appears in the mapping table.
            - The final answer is a single valid JSON object with a "sequence" array.

            ### 8. OUTPUT
            Return ONLY one raw JSON object in this format:
            {{
              "sequence": ["START", "P1", "P3", "D3", "D1"]
            }}
            The robot is assumed to travel to each node in the order you give; do not output
            coordinates, moves, or any other field.
            Do not include explanations or markdown.
            """
        ),
        "image": _clean_prompt(
            """
            You are the route planner for an indoor delivery robot.
            You only receive the BEV indoor map image.

            ### 1. PRIMARY OBJECTIVE
            - Maximize completed task value while minimizing travel time.
            - Prefer a safe, topologically valid node order over an aggressive but risky one.

            ### 2. INDOOR MAP INTERPRETATION
            - START is the green marker.
            - Pickup nodes are red markers labeled P1, P2, P3, ...
            - Delivery nodes are blue markers labeled D1, D2, D3, ...
            - Traversable space is the connected corridor / room layout shown in the image.
            - Black background outside the building is not traversable.
            - Do not cut through walls or blocked regions.
            - Judge node-to-node cost by reachable corridor distance, not by straight-line proximity.

            ### 3. TASK LOGIC
            - Pn must be visited before the matching Dn.
            - Never place Dn before the matching Pn.
            - Each pickup node Pn may appear at most once, and each delivery node Dn may appear at most once.
            - Use a conservative load strategy: avoid long runs of pickups before deliveries.
            - Unless there is a clear visual reason, do not accumulate more than two undelivered pickups before starting deliveries.
            - Prefer short, locally coherent pickup-delivery chains instead of collecting many pickups first.

            ### 4. SELF-CHECK BEFORE YOU ANSWER
            Verify all of the following before returning JSON:
            - Every Dn has an earlier matching Pn.
            - No node is repeated unless it is START.
            - The route is consistent with visible indoor connectivity.
            - The answer is a single valid JSON object with a "sequence" array.

            ### 5. OUTPUT
            Return ONLY one raw JSON object in this format:
            {{
              "sequence": ["START", "P1", "P3", "D3", "D1"]
            }}
            The robot is assumed to travel to each node in the order you give; do not output
            coordinates, moves, or any other field.
            Do not include explanations or markdown.
            """
        ),
    },
    "forest": {
        "meta": _clean_prompt(
            """
            You are the route planner for a delivery robot in a forest scene.
            Generate one strictly sequential visiting order over the task nodes that is globally efficient and fully legal.

            ### 1. PRIMARY OBJECTIVE
            - Maximize total collected reward while minimizing travel time.
            - If rewards are missing or equal, minimize estimated travel time for all feasible tasks.
            - A fully legal plan is better than a shorter but invalid one.

            ### 2. CURRENT STATE
            - Time: {now:.2f} s
            - Position: {{ "x": {cur_x:.2f}, "y": {cur_y:.2f} }}
            - Speed: {speed:.2f} m/s
            - Load: {capacity}/{max_capacity}

            ### 3. VISUAL MAP & ID MAPPING
            - START is the green marker.
            - Pickup nodes are red markers labeled P1, P2, P3, ...
            - Delivery nodes are blue markers labeled D1, D2, D3, ...
            - Task A-0001 corresponds to P1 and D1.
            - Your answer must name these nodes only. The mapping table below tells you where each node is.
            --- MAPPING TABLE START ---
            {id_mapping_section}
            --- MAPPING TABLE END ---

            ### 4. TASK POOL
            Carried tasks:
            {carried_tasks_json}

            Available tasks:
            {new_tasks_json}

            ### 5. FOREST SCENE INTERPRETATION
            - This forest scene is more open than indoor corridors.
            - Large dark free-space regions can often be treated as traversable.
            - Tree-like texture usually acts as weak clutter rather than a strict hard obstacle.
            - Still avoid routes that would obviously pass through dense blocked structures or isolated inaccessible regions.
            - Prefer broad directional sweeps with low backtracking.

            ### 6. HARD CONSTRAINTS
            1. Pn must appear before the matching Dn.
            2. Never place Dn before the matching Pn.
            3. Each pickup node Pn may appear at most once, and each delivery node Dn may appear at most once.
            4. A task that is already carried has no Pn node; only its Dn may be visited.
            5. The number of picked but not yet delivered tasks must never exceed {max_capacity}.
            6. After every Pn, mentally increase load by 1; after every Dn, mentally decrease load by 1.
            7. Use only node labels that exist in the mapping table above. Inventing a node is a planning error.

            ### 7. SELF-CHECK BEFORE YOU ANSWER
            Verify all of the following before returning JSON:
            - Every pickup node appears at most once.
            - Every delivery node appears at most once.
            - Every Dn appears after its matching Pn.
            - Load never exceeds {max_capacity} at any step.
            - Every node label appears in the mapping table.
            - The final answer is a single valid JSON object with a "sequence" array.

            ### 8. OUTPUT
            Return ONLY one raw JSON object in this format:
            {{
              "sequence": ["START", "P1", "P3", "D3", "D1"]
            }}
            The robot is assumed to travel to each node in the order you give; do not output
            coordinates, moves, or any other field.
            Do not include explanations or markdown.
            """
        ),
        "text": _clean_prompt(
            """
            You are the route planner for a delivery robot in a forest scene.
            You only receive structured task text, not the map image.

            ### 1. PRIMARY OBJECTIVE
            - Maximize total collected reward while minimizing estimated travel distance.
            - Use coordinates as the main routing signal.
            - Prefer a fully legal plan over an aggressive but invalid one.

            ### 2. CURRENT STATE
            - Time: {now:.2f} s
            - Position: {{ "x": {cur_x:.2f}, "y": {cur_y:.2f} }}
            - Speed: {speed:.2f} m/s
            - Current load: {capacity}/{max_capacity}

            ### 3. AVAILABLE INFORMATION
            - You do NOT have access to the forest map image.
            - The forest scene is relatively open, so Euclidean distance is often useful.
            - However, larger blocked structures can still create detours.
            - Treat coordinates as a coarse proxy, and prefer smooth global sweeps over repeated backtracking.

            ### 4. TASK TO NODE MAPPING
            - Task A-0001 corresponds to pickup node P1 and delivery node D1.
            - Your answer must name these nodes only. The mapping table below gives their coordinates.
            --- MAPPING TABLE START ---
            {id_mapping_section}
            --- MAPPING TABLE END ---

            ### 5. TASK POOL
            Carried tasks:
            {carried_tasks_json}

            Available tasks:
            {new_tasks_json}

            ### 6. HARD CONSTRAINTS
            1. Pn must appear before the matching Dn.
            2. Never place Dn before the matching Pn.
            3. Each pickup node Pn may appear at most once, and each delivery node Dn may appear at most once.
            4. A task that is already carried has no Pn node; only its Dn may be visited.
            5. The number of picked but not yet delivered tasks must never exceed {max_capacity}.
            6. After every Pn, mentally increase load by 1; after every Dn, mentally decrease load by 1.
            7. Use only node labels that exist in the mapping table above. Inventing a node is a planning error.

            ### 7. SELF-CHECK BEFORE YOU ANSWER
            Verify all of the following before returning JSON:
            - Every pickup node appears at most once.
            - Every delivery node appears at most once.
            - Every Dn appears after its matching Pn.
            - Load never exceeds {max_capacity} at any step.
            - Every node label appears in the mapping table.
            - The final answer is a single valid JSON object with a "sequence" array.

            ### 8. OUTPUT
            Return ONLY one raw JSON object in this format:
            {{
              "sequence": ["START", "P1", "P3", "D3", "D1"]
            }}
            The robot is assumed to travel to each node in the order you give; do not output
            coordinates, moves, or any other field.
            Do not include explanations or markdown.
            """
        ),
        "image": _clean_prompt(
            """
            You are an advanced visual navigation agent for a delivery robot in a forest scene.
            You only receive the map image and must output a safe node sequence.

            ### 1. VISUAL PERCEPTION
            - START is the green marker.
            - Pickup nodes are red markers labeled P1, P2, P3, ...
            - Delivery nodes are blue markers labeled D1, D2, D3, ...
            - Tree-like clutter is usually weak texture, not a hard obstacle.
            - Large blocked structures or clearly separated regions should still be respected.
            - Broad near-straight travel is often reasonable in this scene.

            ### 2. ROUTING LOGIC
            - Prefer globally coherent sweeps over chaotic hopping.
            - Pn must appear before Dn.
            - Each pickup and delivery node may appear at most once.
            - Use a conservative load strategy: do not accumulate long runs of pickups before deliveries.
            - Unless there is a strong reason from the layout, do not keep more than two undelivered pickups pending at once.

            ### 3. SELF-CHECK BEFORE YOU ANSWER
            Verify all of the following before returning JSON:
            - Every Dn has an earlier matching Pn.
            - No node is repeated unless it is START.
            - The sequence is broadly consistent with the visible layout.
            - The answer is a single valid JSON object with a "sequence" array.

            ### 4. OUTPUT
            Return ONLY one raw JSON object in this format:
            {{
              "sequence": ["START", "P1", "P3", "D3", "D1"]
            }}
            The robot is assumed to travel to each node in the order you give; do not output
            coordinates, moves, or any other field.
            Do not include explanations or markdown.
            """
        ),
    },
    "campus": {
        "meta": _clean_prompt(
            """
            You are the route planner for a delivery robot in a campus environment.
            Generate one strictly sequential visiting order over the task nodes that is globally efficient and fully legal.

            ### 1. PRIMARY OBJECTIVE
            - Maximize total collected reward while minimizing travel cost.
            - If rewards are missing or equal, minimize completion time for all feasible tasks.
            - A fully legal plan is better than a shorter but invalid plan.

            ### 2. CURRENT STATE
            - Time: {now:.2f} s
            - Position: {{ "x": {cur_x:.2f}, "y": {cur_y:.2f} }}
            - Speed: {speed:.2f} m/s
            - Load: {capacity}/{max_capacity}

            ### 3. VISUAL MAP & ID MAPPING
            - START is the green marker.
            - Pickup nodes are red markers labeled P1, P2, P3, ...
            - Delivery nodes are blue markers labeled D1, D2, D3, ...
            - Task A-0001 corresponds to P1 and D1.
            - Your answer must name these nodes only. The mapping table below tells you where each node is.
            --- MAPPING TABLE START ---
            {id_mapping_section}
            --- MAPPING TABLE END ---

            ### 4. TASK POOL
            Carried tasks:
            {carried_tasks_json}

            Available tasks:
            {new_tasks_json}

            ### 5. SCENE INTERPRETATION
            - Treat campus as a semi-open outdoor environment.
            - Roads and open plazas may support flexible motion.
            - Large buildings, blocked zones, and disconnected regions should not be crossed blindly.
            - Do not assume fully open Euclidean travel when visible separation exists.
            - Prefer globally coherent sweeps with moderate batching.

            ### 6. HARD CONSTRAINTS
            1. Pn must appear before the matching Dn.
            2. Never place Dn before the matching Pn.
            3. Each pickup node Pn may appear at most once, and each delivery node Dn may appear at most once.
            4. A task that is already carried has no Pn node; only its Dn may be visited.
            5. The number of picked but not yet delivered tasks must never exceed {max_capacity}.
            6. After every Pn, mentally increase load by 1; after every Dn, mentally decrease load by 1.
            7. Use only node labels that exist in the mapping table above. Inventing a node is a planning error.

            ### 7. SELF-CHECK BEFORE YOU ANSWER
            Verify all of the following before returning JSON:
            - Every pickup node appears at most once.
            - Every delivery node appears at most once.
            - Every Dn appears after its matching Pn.
            - Load never exceeds {max_capacity} at any step.
            - Every node label appears in the mapping table.
            - The final answer is a single valid JSON object with a "sequence" array.

            ### 8. OUTPUT
            Return ONLY one raw JSON object in this format:
            {{
              "sequence": ["START", "P1", "P3", "D3", "D1"]
            }}
            The robot is assumed to travel to each node in the order you give; do not output
            coordinates, moves, or any other field.
            Do not include explanations or markdown.
            """
        ),
        "text": _clean_prompt(
            """
            You are the route planner for a delivery robot in a campus environment.
            You only receive structured task text, not the map image.

            ### 1. PRIMARY OBJECTIVE
            - Maximize total collected reward while minimizing estimated travel cost.
            - Use coordinates as the main heuristic, but avoid overly aggressive straight-line assumptions.
            - A fully legal plan is better than a shorter but invalid plan.

            ### 2. CURRENT STATE
            - Time: {now:.2f} s
            - Position: {{ "x": {cur_x:.2f}, "y": {cur_y:.2f} }}
            - Speed: {speed:.2f} m/s
            - Current load: {capacity}/{max_capacity}

            ### 3. AVAILABLE INFORMATION
            - Campus movement is partially constrained.
            - Coordinates are useful but may under-estimate real road distance.
            - Prefer spatially coherent batches and avoid repeated cross-campus zig-zagging.

            ### 4. TASK TO NODE MAPPING
            - Task A-0001 corresponds to pickup node P1 and delivery node D1.
            - Your answer must name these nodes only. The mapping table below gives their coordinates.
            --- MAPPING TABLE START ---
            {id_mapping_section}
            --- MAPPING TABLE END ---

            ### 5. TASK POOL
            Carried tasks:
            {carried_tasks_json}

            Available tasks:
            {new_tasks_json}

            ### 6. HARD CONSTRAINTS
            1. Pn must appear before the matching Dn.
            2. Never place Dn before the matching Pn.
            3. Each pickup node Pn may appear at most once, and each delivery node Dn may appear at most once.
            4. A task that is already carried has no Pn node; only its Dn may be visited.
            5. The number of picked but not yet delivered tasks must never exceed {max_capacity}.
            6. After every Pn, mentally increase load by 1; after every Dn, mentally decrease load by 1.
            7. Use only node labels that exist in the mapping table above. Inventing a node is a planning error.

            ### 7. SELF-CHECK BEFORE YOU ANSWER
            Verify all of the following before returning JSON:
            - Every pickup node appears at most once.
            - Every delivery node appears at most once.
            - Every Dn appears after its matching Pn.
            - Load never exceeds {max_capacity} at any step.
            - Every node label appears in the mapping table.
            - The final answer is a single valid JSON object with a "sequence" array.

            ### 8. OUTPUT
            Return ONLY one raw JSON object in this format:
            {{
              "sequence": ["START", "P1", "P3", "D3", "D1"]
            }}
            The robot is assumed to travel to each node in the order you give; do not output
            coordinates, moves, or any other field.
            Do not include explanations or markdown.
            """
        ),
        "image": _clean_prompt(
            """
            You are an advanced visual navigation agent for a delivery robot in a campus environment.
            You only receive the map image and must output a safe node sequence.

            ### 1. VISUAL PERCEPTION
            - START is the green marker.
            - Pickup nodes are red markers labeled P1, P2, P3, ...
            - Delivery nodes are blue markers labeled D1, D2, D3, ...
            - Use the visible layout to judge which nodes are naturally connected by roads or open plazas.
            - Large buildings, blocked zones, and clearly separated regions should not be crossed blindly.

            ### 2. TASK LOGIC
            - Pn must be visited before the matching Dn.
            - Each pickup and delivery node may appear at most once.
            - Prefer globally coherent routes rather than local greedy hopping.
            - Use a conservative load strategy: avoid collecting many pickups before starting deliveries.
            - Unless the map strongly suggests otherwise, do not keep more than two undelivered pickups pending at once.

            ### 3. SELF-CHECK BEFORE YOU ANSWER
            Verify all of the following before returning JSON:
            - Every Dn has an earlier matching Pn.
            - No node is repeated unless it is START.
            - The sequence is broadly consistent with visible connectivity.
            - The answer is a single valid JSON object with a "sequence" array.

            ### 4. OUTPUT
            Return ONLY one raw JSON object in this format:
            {{
              "sequence": ["START", "P1", "P3", "D3", "D1"]
            }}
            The robot is assumed to travel to each node in the order you give; do not output
            coordinates, moves, or any other field.
            Do not include explanations or markdown.
            """
        ),
    },
    "typical": {
        "meta": _clean_prompt(
            """
            You are the route planner for a delivery robot in a structured open-layout scene.
            Generate a strictly sequential visiting order over the task nodes of the current task set.

            ### 1. PRIMARY OBJECTIVE
            - Maximize total collected reward while minimizing completion time.
            - If rewards are missing or equal, minimize total completion time for all feasible tasks.
            - Travel cost must be estimated from reachable paths in the map, not from raw straight-line distance alone.

            ### 2. CURRENT STATE
            - Time: {now:.2f} s
            - Position: {{ "x": {cur_x:.2f}, "y": {cur_y:.2f} }}
            - Speed: {speed:.2f} m/s
            - Current load: {capacity}/{max_capacity}

            ### 3. SCENE INTERPRETATION
            - The map is a BEV structured environment with broad open lanes and a small number of hard blocked regions.
            - START is the green marker.
            - Pickup nodes are red markers labeled P1, P2, P3, ...
            - Delivery nodes are blue markers labeled D1, D2, D3, ...
            - Broad open space often allows efficient direct motion, but rectangular blocked structures and walls cannot be crossed.
            - If a direct segment would intersect a blocked structure, you must mentally route around it.
            - Do not over-apply indoor corridor assumptions, but do not collapse everything to Euclidean distance either.

            ### 4. TASK TO NODE MAPPING
            - Task A-0001 corresponds to pickup node P1 and delivery node D1.
            - Your answer must name these nodes only. The mapping table below tells you where each node is.
            --- MAPPING TABLE START ---
            {id_mapping_section}
            --- MAPPING TABLE END ---

            ### 5. TASK POOL
            Carried tasks:
            {carried_tasks_json}

            Available tasks:
            {new_tasks_json}

            ### 6. CONSTRAINTS
            1. Pn must appear before the matching Dn.
            2. Never place Dn before the matching Pn.
            3. Each pickup node Pn may appear at most once, and each delivery node Dn may appear at most once.
            4. A task that is already carried has no Pn node; only its Dn may be visited.
            5. The number of picked but not yet delivered tasks must never exceed {max_capacity}.
            6. Use only node labels that exist in the mapping table above. Inventing a node is a planning error.
            7. You may interleave tasks and batch pickups if doing so improves the objective.
            8. Prefer broad, globally coherent sweeps around obstacles rather than local greedy hopping.

            ### 7. SELF-CHECK BEFORE YOU ANSWER
            Verify all of the following before returning JSON:
            - Every pickup node appears at most once.
            - Every delivery node appears at most once.
            - Every Dn appears after its matching Pn.
            - Load never exceeds {max_capacity} at any step.
            - Every node label appears in the mapping table.
            - The final answer is a single valid JSON object with a "sequence" array.

            ### 8. OUTPUT
            Return ONLY one raw JSON object in this format:
            {{
              "sequence": ["START", "P1", "P3", "D3", "D1"]
            }}
            The robot is assumed to travel to each node in the order you give; do not output
            coordinates, moves, or any other field.
            Do not include explanations or markdown.
            """
        ),
        "text": _clean_prompt(
            """
            You are the route planner for a delivery robot in a structured open-layout scene.
            You only receive structured task text, not the map image.

            ### 1. PRIMARY OBJECTIVE
            - Maximize total collected reward while minimizing estimated completion time.
            - If rewards are missing or equal, minimize estimated completion time for all feasible tasks.

            ### 2. CURRENT STATE
            - Time: {now:.2f} s
            - Position: {{ "x": {cur_x:.2f}, "y": {cur_y:.2f} }}
            - Speed: {speed:.2f} m/s
            - Current load: {capacity}/{max_capacity}

            ### 3. AVAILABLE INFORMATION
            - You do NOT have access to the map image.
            - Use the provided coordinates as the main heuristic for travel cost.
            - This scene is more open than indoor, so some direct motion is plausible.
            - However, sparse blocked structures can still force detours, so avoid assuming pure Euclidean travel everywhere.

            ### 4. TASK TO NODE MAPPING
            - Task A-0001 corresponds to pickup node P1 and delivery node D1.
            - Your answer must name these nodes only. The mapping table below gives their coordinates.
            --- MAPPING TABLE START ---
            {id_mapping_section}
            --- MAPPING TABLE END ---

            ### 5. TASK POOL
            Carried tasks:
            {carried_tasks_json}

            Available tasks:
            {new_tasks_json}

            ### 6. CONSTRAINTS
            1. Pn must appear before the matching Dn.
            2. Never place Dn before the matching Pn.
            3. Each pickup node Pn may appear at most once, and each delivery node Dn may appear at most once.
            4. A task that is already carried has no Pn node; only its Dn may be visited.
            5. The number of picked but not yet delivered tasks must never exceed {max_capacity}.
            6. Use only node labels that exist in the mapping table above. Inventing a node is a planning error.
            7. You may batch pickups and reorder deliveries if feasible.
            8. Produce a robust route, not just a greedy nearest-point list.

            ### 7. SELF-CHECK BEFORE YOU ANSWER
            Verify all of the following before returning JSON:
            - Every pickup node appears at most once.
            - Every delivery node appears at most once.
            - Every Dn appears after its matching Pn.
            - Load never exceeds {max_capacity} at any step.
            - Every node label appears in the mapping table.
            - The final answer is a single valid JSON object with a "sequence" array.

            ### 8. OUTPUT
            Return ONLY one raw JSON object in this format:
            {{
              "sequence": ["START", "P1", "P3", "D3", "D1"]
            }}
            The robot is assumed to travel to each node in the order you give; do not output
            coordinates, moves, or any other field.
            Do not include explanations or markdown.
            """
        ),
        "image": _clean_prompt(
            """
            You are the route planner for a delivery robot in a structured open-layout scene.
            You only receive the BEV map image.

            ### 1. PRIMARY OBJECTIVE
            - Maximize total completed task value while minimizing travel time.
            - If rewards are not visible, minimize total completion time for all visible task pairs.

            ### 2. VISUAL MAP INTERPRETATION
            - START is the green marker.
            - Pickup nodes are red markers labeled P1, P2, P3, ...
            - Delivery nodes are blue markers labeled D1, D2, D3, ...
            - The scene has broad traversable space with a small number of hard blocked regions.
            - Near-straight travel is often reasonable when no obstacle lies in the way.
            - Do not cut through walls or rectangular blocked structures.
            - Estimate node-to-node cost by reachable path around visible obstacles, not by raw straight-line proximity alone.

            ### 3. TASK LOGIC
            - Pn must be visited before the matching Dn.
            - Never place Dn before the matching Pn in the sequence.
            - Each pickup node Pn may appear at most once, and each delivery node Dn may appear at most once.
            - You may interleave multiple tasks if that reduces total travel time.
            - Prefer broad sweeps and low-backtracking orderings.

            ### 4. SELF-CHECK BEFORE YOU ANSWER
            Verify all of the following before returning JSON:
            - Every Dn has an earlier matching Pn.
            - No node is repeated unless it is START.
            - The answer is a single valid JSON object with a "sequence" array.

            ### 5. OUTPUT
            Return ONLY one raw JSON object in this format:
            {{
              "sequence": ["START", "P1", "P3", "D3", "D1"]
            }}
            The robot is assumed to travel to each node in the order you give; do not output
            coordinates, moves, or any other field.
            Do not include explanations or markdown.
            """
        ),
    },
    "typical_islands": {
        "meta": _clean_prompt(
            """
            You are the route planner for a delivery robot in an open scene with island obstacles.
            Generate a strictly sequential visiting order over the task nodes of the current task set.

            ### 1. PRIMARY OBJECTIVE
            - Maximize total collected reward while minimizing completion time.
            - If rewards are missing or equal, minimize total completion time for all feasible tasks.
            - Travel cost must be estimated from reachable free-space routes around obstacle islands, not from straight-line distance.

            ### 2. CURRENT STATE
            - Time: {now:.2f} s
            - Position: {{ "x": {cur_x:.2f}, "y": {cur_y:.2f} }}
            - Speed: {speed:.2f} m/s
            - Current load: {capacity}/{max_capacity}

            ### 3. SCENE INTERPRETATION
            - The map is a BEV open area containing several isolated hard obstacles that act like islands.
            - START is the green marker.
            - Pickup nodes are red markers labeled P1, P2, P3, ...
            - Delivery nodes are blue markers labeled D1, D2, D3, ...
            - Free space between islands is traversable and often supports broad sweeps.
            - You must never cut through an island obstacle.
            - If a direct segment would pass through an island, you must mentally detour around it through the visible gaps.
            - Judge route quality by true reachable detours around islands, not by Euclidean distance.

            ### 4. TASK TO NODE MAPPING
            - Task A-0001 corresponds to pickup node P1 and delivery node D1.
            - Your answer must name these nodes only. The mapping table below tells you where each node is.
            --- MAPPING TABLE START ---
            {id_mapping_section}
            --- MAPPING TABLE END ---

            ### 5. TASK POOL
            Carried tasks:
            {carried_tasks_json}

            Available tasks:
            {new_tasks_json}

            ### 6. CONSTRAINTS
            1. Pn must appear before the matching Dn.
            2. Never place Dn before the matching Pn.
            3. Each pickup node Pn may appear at most once, and each delivery node Dn may appear at most once.
            4. A task that is already carried has no Pn node; only its Dn may be visited.
            5. The number of picked but not yet delivered tasks must never exceed {max_capacity}.
            6. Use only node labels that exist in the mapping table above. Inventing a node is a planning error.
            7. You may interleave tasks and batch pickups if doing so improves the objective.
            8. Prefer routes that exploit open channels around islands while avoiding unnecessary cross-map zig-zagging.

            ### 7. SELF-CHECK BEFORE YOU ANSWER
            Verify all of the following before returning JSON:
            - Every pickup node appears at most once.
            - Every delivery node appears at most once.
            - Every Dn appears after its matching Pn.
            - Load never exceeds {max_capacity} at any step.
            - Every node label appears in the mapping table.
            - The final answer is a single valid JSON object with a "sequence" array.

            ### 8. OUTPUT
            Return ONLY one raw JSON object in this format:
            {{
              "sequence": ["START", "P1", "P3", "D3", "D1"]
            }}
            The robot is assumed to travel to each node in the order you give; do not output
            coordinates, moves, or any other field.
            Do not include explanations or markdown.
            """
        ),
        "text": _clean_prompt(
            """
            You are the route planner for a delivery robot in an open scene with island obstacles.
            You only receive structured task text, not the map image.

            ### 1. PRIMARY OBJECTIVE
            - Maximize total collected reward while minimizing estimated completion time.
            - If rewards are missing or equal, minimize estimated completion time for all feasible tasks.

            ### 2. CURRENT STATE
            - Time: {now:.2f} s
            - Position: {{ "x": {cur_x:.2f}, "y": {cur_y:.2f} }}
            - Speed: {speed:.2f} m/s
            - Current load: {capacity}/{max_capacity}

            ### 3. AVAILABLE INFORMATION
            - You do NOT have access to the map image.
            - Use the provided coordinates as the main heuristic for travel cost.
            - This scene is fairly open, but isolated obstacles may still force significant detours around them.
            - Avoid assuming that every pair of points can be connected by a direct straight path.

            ### 4. TASK TO NODE MAPPING
            - Task A-0001 corresponds to pickup node P1 and delivery node D1.
            - Your answer must name these nodes only. The mapping table below gives their coordinates.
            --- MAPPING TABLE START ---
            {id_mapping_section}
            --- MAPPING TABLE END ---

            ### 5. TASK POOL
            Carried tasks:
            {carried_tasks_json}

            Available tasks:
            {new_tasks_json}

            ### 6. CONSTRAINTS
            1. Pn must appear before the matching Dn.
            2. Never place Dn before the matching Pn.
            3. Each pickup node Pn may appear at most once, and each delivery node Dn may appear at most once.
            4. A task that is already carried has no Pn node; only its Dn may be visited.
            5. The number of picked but not yet delivered tasks must never exceed {max_capacity}.
            6. Use only node labels that exist in the mapping table above. Inventing a node is a planning error.
            7. You may batch pickups and reorder deliveries if feasible.
            8. Produce a robust route, not just a greedy nearest-point list.

            ### 7. SELF-CHECK BEFORE YOU ANSWER
            Verify all of the following before returning JSON:
            - Every pickup node appears at most once.
            - Every delivery node appears at most once.
            - Every Dn appears after its matching Pn.
            - Load never exceeds {max_capacity} at any step.
            - Every node label appears in the mapping table.
            - The final answer is a single valid JSON object with a "sequence" array.

            ### 8. OUTPUT
            Return ONLY one raw JSON object in this format:
            {{
              "sequence": ["START", "P1", "P3", "D3", "D1"]
            }}
            The robot is assumed to travel to each node in the order you give; do not output
            coordinates, moves, or any other field.
            Do not include explanations or markdown.
            """
        ),
        "image": _clean_prompt(
            """
            You are the route planner for a delivery robot in an open scene with island obstacles.
            You only receive the BEV map image.

            ### 1. PRIMARY OBJECTIVE
            - Maximize total completed task value while minimizing travel time.
            - If rewards are not visible, minimize total completion time for all visible task pairs.

            ### 2. VISUAL MAP INTERPRETATION
            - START is the green marker.
            - Pickup nodes are red markers labeled P1, P2, P3, ...
            - Delivery nodes are blue markers labeled D1, D2, D3, ...
            - The scene contains isolated obstacle islands inside otherwise open traversable space.
            - You must not cut through any island obstacle.
            - Near-straight travel is reasonable only when the segment remains in free space and does not intersect an island.
            - Estimate node-to-node cost by reachable routes through the visible gaps around islands.

            ### 3. TASK LOGIC
            - Pn must be visited before the matching Dn.
            - Never place Dn before the matching Pn in the sequence.
            - Each pickup node Pn may appear at most once, and each delivery node Dn may appear at most once.
            - You may interleave multiple tasks if that reduces total travel time.
            - Prefer routes that sweep around islands efficiently instead of repeatedly crossing the map.

            ### 4. SELF-CHECK BEFORE YOU ANSWER
            Verify all of the following before returning JSON:
            - Every Dn has an earlier matching Pn.
            - No node is repeated unless it is START.
            - The answer is a single valid JSON object with a "sequence" array.

            ### 5. OUTPUT
            Return ONLY one raw JSON object in this format:
            {{
              "sequence": ["START", "P1", "P3", "D3", "D1"]
            }}
            The robot is assumed to travel to each node in the order you give; do not output
            coordinates, moves, or any other field.
            Do not include explanations or markdown.
            """
        ),
    },
}


def get_planner_prompt(scene: str, planner_type: str) -> str:
    # Every released scene has an active override. Resolve that registry
    # directly instead of requiring a legacy JSON file that is never selected.
    resolved_scene = scene if scene in SCENE_PROMPT_OVERRIDES else PROMPT_SCENE_ALIASES.get(scene, scene)
    scene_prompts = SCENE_PROMPT_OVERRIDES.get(resolved_scene)
    if not scene_prompts:
        raise ValueError(f"未找到 scene={scene} 的 prompt 配置")

    prompt = scene_prompts.get(planner_type)
    if not prompt:
        raise ValueError(f"scene={scene} 缺少 planner_type={planner_type} 的 prompt 配置")

    return prompt
