import time
from heuristics import manhattan
from space_time_astar import space_time_astar

def detect_collisions(paths):
    """
    Detect all collisions between paths.
    
    Returns:
        list of collision dicts: {'agents': (i, j), 'type': 'vertex'/'edge', 'loc': ..., 'time': t}
    """
    collisions = []
    if not paths:
        return collisions
    
    max_t = max(len(p) for p in paths)
    
    for t in range(max_t):
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                # Get positions (agents stay at goal after reaching it)
                pos_i = paths[i][min(t, len(paths[i]) - 1)]
                pos_j = paths[j][min(t, len(paths[j]) - 1)]
                
                # Vertex collision
                if pos_i == pos_j:
                    collisions.append({
                        'agents': (i, j),
                        'type': 'vertex',
                        'loc': pos_i,
                        'time': t
                    })
                
                # Edge collision (swap)
                if t > 0:
                    prev_i = paths[i][min(t - 1, len(paths[i]) - 1)]
                    prev_j = paths[j][min(t - 1, len(paths[j]) - 1)]
                    if pos_i == prev_j and pos_j == prev_i:
                        collisions.append({
                            'agents': (i, j),
                            'type': 'edge',
                            'loc': (prev_i, pos_i),
                            'time': t
                        })
    
    return collisions


def independent_astar(grid, starts, goals, heuristic=None):
    """
    Run independent A* for each agent (ignoring other agents).
    Paths may have collisions.
    
    Returns:
        dict with 'paths', 'sum_of_costs', 'makespan', 'collisions', 'time', 'success'
    """
    if heuristic is None:
        heuristic = manhattan
    
    start_time = time.perf_counter()
    paths = []
    total_cost = 0
    success = True
    
    for i in range(len(starts)):
        result = space_time_astar(grid, starts[i], goals[i], heuristic)
        if result['path'] is None:
            success = False
            paths.append([starts[i]])  # Stay at start
        else:
            paths.append(result['path'])
            total_cost += result['cost']
    
    collisions = detect_collisions(paths)
    makespan = max(len(p) - 1 for p in paths) if paths else 0
    elapsed = time.perf_counter() - start_time
    
    return {
        'paths': paths,
        'sum_of_costs': total_cost,
        'makespan': makespan,
        'collisions': len(collisions),
        'collision_list': collisions,
        'time': elapsed,
        'success': success
    }


def prioritized_planning(grid, starts, goals, heuristic=None):
    """
    Prioritized Planning: plan agents sequentially with reservations.
    Higher priority agents' paths become constraints for lower priority agents.
    
    Priority order: agents with longer individual shortest paths go first.
    
    Returns:
        dict with 'paths', 'sum_of_costs', 'makespan', 'collisions', 'time', 'success'
    """
    if heuristic is None:
        heuristic = manhattan
    
    start_time = time.perf_counter()
    num_agents = len(starts)
    
    # Determine priority order: sort by manhattan distance to goal (descending)
    priorities = sorted(range(num_agents),
                       key=lambda i: manhattan(starts[i], goals[i]),
                       reverse=True)
    
    paths = [None] * num_agents
    constraints = []
    success = True
    total_cost = 0
    
    for agent_idx in priorities:
        # Build constraints from previously planned agents' paths
        agent_constraints = []
        for prev_agent in priorities:
            if paths[prev_agent] is None:
                continue
            prev_path = paths[prev_agent]
            max_t = len(prev_path)
            
            for t in range(max_t):
                # Vertex constraint: can't be where prev_agent is at time t
                agent_constraints.append({
                    'agent': agent_idx,
                    'loc': [prev_path[t]],
                    'time': t
                })
            
            # Agent stays at goal after reaching it
            # Add constraints for goal position for extra time steps
            goal_pos = prev_path[-1]
            for t in range(max_t, max_t + grid.width + grid.height):
                agent_constraints.append({
                    'agent': agent_idx,
                    'loc': [goal_pos],
                    'time': t
                })
            
            # Edge constraints (swap prevention)
            for t in range(len(prev_path) - 1):
                agent_constraints.append({
                    'agent': agent_idx,
                    'loc': [prev_path[t + 1], prev_path[t]],
                    'time': t + 1
                })
        
        result = space_time_astar(grid, starts[agent_idx], goals[agent_idx],
                                  heuristic, agent_constraints)
        
        if result['path'] is None:
            success = False
            paths[agent_idx] = [starts[agent_idx]]  # Stay at start
        else:
            paths[agent_idx] = result['path']
            total_cost += result['cost']
    
    collisions = detect_collisions(paths)
    makespan = max(len(p) - 1 for p in paths) if paths else 0
    elapsed = time.perf_counter() - start_time
    
    return {
        'paths': paths,
        'sum_of_costs': total_cost,
        'makespan': makespan,
        'collisions': len(collisions),
        'collision_list': collisions,
        'time': elapsed,
        'success': success
    }
