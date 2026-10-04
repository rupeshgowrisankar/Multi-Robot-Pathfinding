import heapq
import time
import copy
from heuristics import manhattan
from space_time_astar import space_time_astar
from prioritized_planning import detect_collisions


class CTNode:
    """Conflict Tree node for CBS."""
    def __init__(self):
        self.constraints = []   # List of constraint dicts
        self.paths = []         # List of paths (one per agent)
        self.cost = 0           # Sum of costs
    
    def __lt__(self, other):
        return self.cost < other.cost


def detect_first_collision(paths):
    """
    Detect the FIRST collision between any pair of agents.
    Returns collision dict or None.
    """
    if not paths:
        return None
    
    max_t = max(len(p) for p in paths)
    
    for t in range(max_t):
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                pos_i = paths[i][min(t, len(paths[i]) - 1)]
                pos_j = paths[j][min(t, len(paths[j]) - 1)]
                
                # Vertex collision
                if pos_i == pos_j:
                    return {
                        'agents': (i, j),
                        'type': 'vertex',
                        'loc': [pos_i],
                        'time': t
                    }
                
                # Edge collision (swap)
                if t > 0:
                    prev_i = paths[i][min(t - 1, len(paths[i]) - 1)]
                    prev_j = paths[j][min(t - 1, len(paths[j]) - 1)]
                    if pos_i == prev_j and pos_j == prev_i:
                        return {
                            'agents': (i, j),
                            'type': 'edge',
                            'loc': [prev_i, pos_i],
                            'time': t
                        }
    
    return None


def cbs(grid, starts, goals, heuristic=None, time_limit=30):
    """
    Conflict-Based Search — optimal multi-agent pathfinding.
    
    High-level: searches Conflict Tree (CT) using best-first search on sum of costs.
    Low-level: Space-Time A* for individual agents under constraints.
    
    Args:
        grid: Grid object
        starts: list of (x, y) start positions
        goals: list of (x, y) goal positions  
        heuristic: heuristic function (default: manhattan)
        time_limit: max seconds before timeout
    
    Returns:
        dict with 'paths', 'sum_of_costs', 'makespan', 'ct_nodes_expanded',
                  'time', 'success'
    """
    if heuristic is None:
        heuristic = manhattan
    
    start_time = time.perf_counter()
    num_agents = len(starts)
    
    # Initialize root CT node
    root = CTNode()
    root.constraints = []
    root.paths = []
    
    for i in range(num_agents):
        result = space_time_astar(grid, starts[i], goals[i], heuristic)
        if result['path'] is None:
            elapsed = time.perf_counter() - start_time
            return {
                'paths': [[s] for s in starts],
                'sum_of_costs': 0,
                'makespan': 0,
                'ct_nodes_expanded': 0,
                'time': elapsed,
                'success': False
            }
        root.paths.append(result['path'])
    
    root.cost = sum(len(p) - 1 for p in root.paths)
    
    # High-level search
    open_list = []
    counter = 0
    heapq.heappush(open_list, (root.cost, counter, root))
    ct_nodes_expanded = 0
    
    while open_list:
        # Check time limit
        if time.perf_counter() - start_time > time_limit:
            break
        
        _, _, current_node = heapq.heappop(open_list)
        ct_nodes_expanded += 1
        
        # Find first collision
        collision = detect_first_collision(current_node.paths)
        
        if collision is None:
            # No collisions — optimal solution found!
            elapsed = time.perf_counter() - start_time
            paths = current_node.paths
            collisions_check = detect_collisions(paths)
            return {
                'paths': paths,
                'sum_of_costs': current_node.cost,
                'makespan': max(len(p) - 1 for p in paths),
                'ct_nodes_expanded': ct_nodes_expanded,
                'time': elapsed,
                'success': True
            }
        
        # Branch: create two child nodes
        for agent_idx in collision['agents']:
            child = CTNode()
            child.constraints = copy.deepcopy(current_node.constraints)
            
            # Add new constraint for this agent
            new_constraint = {
                'agent': agent_idx,
                'loc': collision['loc'],
                'time': collision['time']
            }
            child.constraints.append(new_constraint)
            
            # Copy paths from parent
            child.paths = [p[:] for p in current_node.paths]
            
            # Re-plan only for the constrained agent
            agent_constraints = [c for c in child.constraints if c['agent'] == agent_idx]
            
            result = space_time_astar(
                grid, starts[agent_idx], goals[agent_idx],
                heuristic, agent_constraints
            )
            
            if result['path'] is not None:
                child.paths[agent_idx] = result['path']
                child.cost = sum(len(p) - 1 for p in child.paths)
                counter += 1
                heapq.heappush(open_list, (child.cost, counter, child))
    
    # Timeout or no solution
    elapsed = time.perf_counter() - start_time
    return {
        'paths': [[s] for s in starts],
        'sum_of_costs': 0,
        'makespan': 0,
        'ct_nodes_expanded': ct_nodes_expanded,
        'time': elapsed,
        'success': False
    }
