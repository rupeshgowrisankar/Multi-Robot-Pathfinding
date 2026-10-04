import heapq
import time

def astar(grid, start, goal, heuristic):
    """
    A* search algorithm for single-agent pathfinding.
    
    Args:
        grid: Grid object with get_neighbors() method
        start: (x, y) start position
        goal: (x, y) goal position
        heuristic: function(a, b) -> float, the heuristic function
    
    Returns:
        dict with keys:
            'path': list of (x, y) from start to goal, or None if no path
            'cost': path cost (number of steps), or -1 if no path
            'nodes_expanded': number of nodes popped from open list
            'nodes_generated': number of nodes added to open list
            'time': execution time in seconds
    """
    start_time = time.perf_counter()
    
    # Priority queue: (f_score, tie_breaker_h, counter, node)
    # Using h as tie-breaker: prefer nodes closer to goal when f is equal
    counter = 0
    open_list = []
    h_start = heuristic(start, goal)
    heapq.heappush(open_list, (h_start, h_start, counter, start))
    
    came_from = {}
    g_score = {start: 0}
    closed_set = set()
    
    nodes_expanded = 0
    nodes_generated = 1  # start node
    
    while open_list:
        f, h, _, current = heapq.heappop(open_list)
        
        if current in closed_set:
            continue
        
        nodes_expanded += 1
        closed_set.add(current)
        
        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            
            elapsed = time.perf_counter() - start_time
            return {
                'path': path,
                'cost': g_score[goal],
                'nodes_expanded': nodes_expanded,
                'nodes_generated': nodes_generated,
                'time': elapsed
            }
        
        for neighbor in grid.get_neighbors(current):
            if neighbor in closed_set:
                continue
            
            tentative_g = g_score[current] + 1  # uniform cost grid
            
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h_val = heuristic(neighbor, goal)
                f_val = tentative_g + h_val
                counter += 1
                nodes_generated += 1
                heapq.heappush(open_list, (f_val, h_val, counter, neighbor))
    
    elapsed = time.perf_counter() - start_time
    return {
        'path': None,
        'cost': -1,
        'nodes_expanded': nodes_expanded,
        'nodes_generated': nodes_generated,
        'time': elapsed
    }
