"""
Hybrid Prioritized Planning + Conflict-Based Search (H-PCBS)

Proposed algorithm:
    Stage 1: Prioritized Planning creates a fast initial solution.
    Stage 2: If PP fails or leaves conflicts, CBS-style conflict resolution
             starts from the PP solution and replans only conflicting agents.

This combines the speed of PP with the systematic conflict resolution of CBS.
"""

import heapq
import time
import copy

from heuristics import manhattan
from space_time_astar import space_time_astar
from prioritized_planning import prioritized_planning, detect_collisions
from cbs import detect_first_collision


class HybridCTNode:
    def __init__(self):
        self.constraints = []
        self.paths = []
        self.cost = 0

    def __lt__(self, other):
        return self.cost < other.cost


def hybrid_pp_cbs(grid, starts, goals, heuristic=None, time_limit=10):
    """Run PP first, then localized CBS repair if necessary."""
    if heuristic is None:
        heuristic = manhattan

    total_start = time.perf_counter()

    # =========================================================
    # STAGE 1: PRIORITIZED PLANNING
    # =========================================================
    pp_start = time.perf_counter()
    pp = prioritized_planning(grid, starts, goals, heuristic)
    pp_time = time.perf_counter() - pp_start

    initial_pp_success = pp["success"]
    initial_collisions = pp["collisions"]

    # Fast path: PP already solved the instance.
    if initial_pp_success and initial_collisions == 0:
        return {
            "paths": pp["paths"],
            "sum_of_costs": pp["sum_of_costs"],
            "makespan": pp["makespan"],
            "collisions": 0,
            "time": time.perf_counter() - total_start,
            "success": True,
            "initial_pp_success": True,
            "initial_collisions": 0,
            "ct_nodes_expanded": 0,
            "replans": 0,
            "pp_time": pp_time,
            "repair_time": 0.0,
            "used_repair": False
        }

    # =========================================================
    # STAGE 2: CREATE A FEASIBLE PP-SEEDED ROOT
    # =========================================================
    # Keep successful PP paths. If PP failed for an agent, obtain
    # an unconstrained feasible path so CBS has a complete root.
    seed_paths = [p[:] if p else None for p in pp["paths"]]

    for i in range(len(seed_paths)):
        if pp["success"] and seed_paths[i] is not None:
            continue

        result = space_time_astar(
            grid, starts[i], goals[i], heuristic
        )

        if result["path"] is None:
            return {
                "paths": seed_paths,
                "sum_of_costs": 0,
                "makespan": 0,
                "collisions": initial_collisions,
                "time": time.perf_counter() - total_start,
                "success": False,
                "initial_pp_success": initial_pp_success,
                "initial_collisions": initial_collisions,
                "ct_nodes_expanded": 0,
                "replans": 0,
                "pp_time": pp_time,
                "repair_time": time.perf_counter() - pp_start,
                "used_repair": True
            }

        seed_paths[i] = result["path"]

    # =========================================================
    # STAGE 3: CBS-STYLE LOCALIZED REPAIR
    # =========================================================
    repair_start = time.perf_counter()

    root = HybridCTNode()
    root.paths = seed_paths
    root.cost = sum(len(p) - 1 for p in root.paths)

    open_list = []
    counter = 0
    heapq.heappush(open_list, (root.cost, counter, root))

    ct_nodes_expanded = 0
    replans = 0
    deadline = total_start + time_limit

    while open_list and time.perf_counter() <= deadline:
        _, _, current = heapq.heappop(open_list)
        ct_nodes_expanded += 1

        collision = detect_first_collision(current.paths)

        # Collision-free solution found.
        if collision is None:
            return {
                "paths": current.paths,
                "sum_of_costs": current.cost,
                "makespan": max(len(p) - 1 for p in current.paths),
                "collisions": 0,
                "time": time.perf_counter() - total_start,
                "success": True,
                "initial_pp_success": initial_pp_success,
                "initial_collisions": initial_collisions,
                "ct_nodes_expanded": ct_nodes_expanded,
                "replans": replans,
                "pp_time": pp_time,
                "repair_time": time.perf_counter() - repair_start,
                "used_repair": True
            }

        # CBS branch: constrain each of the two conflicting agents.
        for idx, agent_idx in enumerate(collision["agents"]):
            if time.perf_counter() > deadline:
                break

            child = HybridCTNode()
            child.constraints = copy.deepcopy(current.constraints)
            child.paths = [p[:] for p in current.paths]

            if collision["type"] == "edge":
                loc = collision["loc"] if idx == 0 else [collision["loc"][1], collision["loc"][0]]
            else:
                loc = collision["loc"]

            child.constraints.append({
                "agent": agent_idx,
                "loc": loc,
                "time": collision["time"]
            })

            agent_constraints = [
                c for c in child.constraints
                if c["agent"] == agent_idx
            ]

            result = space_time_astar(
                grid,
                starts[agent_idx],
                goals[agent_idx],
                heuristic,
                agent_constraints
            )
            replans += 1

            if result["path"] is not None:
                child.paths[agent_idx] = result["path"]
                child.cost = sum(len(p) - 1 for p in child.paths)

                counter += 1
                heapq.heappush(
                    open_list,
                    (child.cost, counter, child)
                )

    # Repair failed within the time limit.
    remaining_collisions = len(detect_collisions(root.paths))

    return {
        "paths": root.paths,
        "sum_of_costs": root.cost,
        "makespan": max(len(p) - 1 for p in root.paths),
        "collisions": remaining_collisions,
        "time": time.perf_counter() - total_start,
        "success": False,
        "initial_pp_success": initial_pp_success,
        "initial_collisions": initial_collisions,
        "ct_nodes_expanded": ct_nodes_expanded,
        "replans": replans,
        "pp_time": pp_time,
        "repair_time": time.perf_counter() - repair_start,
        "used_repair": True
    }


hpcbs = hybrid_pp_cbs
