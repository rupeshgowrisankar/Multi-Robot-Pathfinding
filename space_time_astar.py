import heapq
import time
from heuristics import manhattan

def space_time_astar(grid, start, goal, heuristic=None, constraints=None, max_time=None):
    """
    Space-Time A* search for single agent with constraints.
    State: (x, y, t) — position at a specific time step.
    
    Args:
        grid: Grid object
        start: (x, y) start position
        goal: (x, y) goal position
        heuristic: heuristic function (default: manhattan)
        constraints: list of constraint dicts:
            Vertex: {'agent': int, 'loc': [(x, y)], 'time': t}
            Edge:   {'agent': int, 'loc': [(x1,y1), (x2,y2)], 'time': t}
        max_time: maximum time steps allowed (default: width * height)
    
    Returns:
        dict with 'path', 'cost', 'nodes_expanded', 'time'
        path is list of (x, y) positions at each time step
    """
    if heuristic is None:
        heuristic = manhattan
    if constraints is None:
        constraints = []
    if max_time is None:
        max_time = grid.width * grid.height
    
    start_time_clock = time.perf_counter()
    
    # Build constraint tables for O(1) lookup
    # Vertex constraints: {(x, y, t)} — agent cannot be at (x,y) at time t
    vertex_constraints = set()
    # Edge constraints: {(x1, y1, x2, y2, t)} — agent cannot move from (x1,y1) to (x2,y2) at time t
    edge_constraints = set()
    
    for c in constraints:
        if len(c['loc']) == 1:
            vertex_constraints.add((c['loc'][0][0], c['loc'][0][1], c['time']))
        elif len(c['loc']) == 2:
            edge_constraints.add((c['loc'][0][0], c['loc'][0][1],
                                  c['loc'][1][0], c['loc'][1][1], c['time']))
    
    def is_constrained(curr, next_pos, next_time):
        """Check if moving to next_pos at next_time violates any constraint."""
        # Vertex constraint
        if (next_pos[0], next_pos[1], next_time) in vertex_constraints:
            return True
        # Edge constraint
        if (curr[0], curr[1], next_pos[0], next_pos[1], next_time) in edge_constraints:
            return True
        return False
    
    # State: (x, y, t)
    start_state = (start[0], start[1], 0)
    h_val = heuristic(start, goal)
    
    counter = 0
    open_list = [(h_val, h_val, counter, start_state)]
    
    came_from = {}
    g_score = {start_state: 0}
    closed_set = set()
    nodes_expanded = 0
    
    while open_list:
        f, h, _, current = heapq.heappop(open_list)
        
        if current in closed_set:
            continue
        
        nodes_expanded += 1
        closed_set.add(current)
        
        cx, cy, ct = current
        
        # Goal check: reached goal position
        if (cx, cy) == goal:
            # Check no future vertex constraints at goal
            has_future_constraint = False
            for vc_x, vc_y, vc_t in vertex_constraints:
                if (vc_x, vc_y) == goal and vc_t > ct:
                    has_future_constraint = True
                    break
            
            if not has_future_constraint:
                # Reconstruct path
                path = []
                state = current
                while state in came_from:
                    path.append((state[0], state[1]))
                    state = came_from[state]
                path.append(start)
                path.reverse()
                
                elapsed = time.perf_counter() - start_time_clock
                return {
                    'path': path,
                    'cost': len(path) - 1,
                    'nodes_expanded': nodes_expanded,
                    'time': elapsed
                }
        
        if ct >= max_time:
            continue
        
        # Generate successors: 4 moves + 1 wait
        neighbors = grid.get_neighbors((cx, cy))
        # Add wait action
        moves = [(cx, cy)] + neighbors
        
        for nx, ny in moves:
            next_time = ct + 1
            next_state = (nx, ny, next_time)
            
            if next_state in closed_set:
                continue
            
            if is_constrained((cx, cy), (nx, ny), next_time):
                continue
            
            tentative_g = g_score[current] + 1
            
            if tentative_g < g_score.get(next_state, float('inf')):
                came_from[next_state] = current
                g_score[next_state] = tentative_g
                h_val = heuristic((nx, ny), goal)
                f_val = tentative_g + h_val
                counter += 1
                heapq.heappush(open_list, (f_val, h_val, counter, next_state))
    
    elapsed = time.perf_counter() - start_time_clock
    return {
        'path': None,
        'cost': -1,
        'nodes_expanded': nodes_expanded,
        'time': elapsed
    }
