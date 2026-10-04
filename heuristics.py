import math

def manhattan(a, b):
    """Manhattan distance (L1 norm). Optimal for 4-connected grids."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    """Euclidean distance (L2 norm). Always admissible but underestimates on 4-connected grids."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def chebyshev(a, b):
    """Chebyshev distance (L-infinity norm). Optimal for 8-connected grids with uniform cost."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def zero(a, b):
    """Zero heuristic. Reduces A* to Dijkstra's algorithm."""
    return 0

# Dictionary mapping names to functions for easy iteration
HEURISTICS = {
    'Manhattan': manhattan,
    'Euclidean': euclidean,
    'Chebyshev': chebyshev,
    'Zero (Dijkstra)': zero
}
