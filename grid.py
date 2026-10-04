import random
from collections import deque

class Grid:
    """2D grid with random obstacles for pathfinding."""
    
    def __init__(self, width, height, obstacle_density=0.2, seed=None):
        """
        Create a grid with random obstacles.
        
        Args:
            width: Grid width (columns)
            height: Grid height (rows) 
            obstacle_density: Fraction of cells that are obstacles (0.0 to 0.4)
            seed: Random seed for reproducibility
        """
        # Store params
        self.width = width
        self.height = height
        self.obstacle_density = obstacle_density
        
        if seed is not None:
            random.seed(seed)
        
        # Generate obstacles
        self.obstacles = set()
        total_cells = width * height
        num_obstacles = int(total_cells * obstacle_density)
        
        all_cells = [(x, y) for x in range(width) for y in range(height)]
        random.shuffle(all_cells)
        
        # Place obstacles (we'll remove start/goal later)
        self.obstacles = set(all_cells[:num_obstacles])
        
        # Default start and goal
        self.start = (0, 0)
        self.goal = (width - 1, height - 1)
        
        # Ensure start and goal are not obstacles
        self.obstacles.discard(self.start)
        self.obstacles.discard(self.goal)
        
        # Ensure path exists using BFS; if not, regenerate
        # Keep regenerating until a valid grid is found
        attempts = 0
        while not self._path_exists(self.start, self.goal):
            attempts += 1
            if attempts > 100:
                # Reduce obstacle density
                self.obstacles = set()
                num_obstacles = max(0, num_obstacles - 5)
            random.shuffle(all_cells)
            self.obstacles = set(all_cells[:num_obstacles])
            self.obstacles.discard(self.start)
            self.obstacles.discard(self.goal)
    
    def is_valid(self, cell):
        """Check if a cell is within bounds and not an obstacle."""
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height and cell not in self.obstacles
    
    def get_neighbors(self, cell):
        """Get valid 4-connected neighbors (up, down, left, right)."""
        x, y = cell
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self.is_valid((nx, ny)):
                neighbors.append((nx, ny))
        return neighbors
    
    def _path_exists(self, start, goal):
        """BFS to check if a path exists between start and goal."""
        if not self.is_valid(start) or not self.is_valid(goal):
            return False
        visited = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current == goal:
                return True
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False
    
    def set_start_goal(self, start, goal):
        """Set custom start and goal positions."""
        self.start = start
        self.goal = goal
        self.obstacles.discard(start)
        self.obstacles.discard(goal)
    
    @staticmethod
    def generate_multi_agent(width, height, num_agents, obstacle_density=0.2, seed=None):
        """
        Generate a grid with multiple start/goal pairs.
        All starts and goals are distinct and reachable.
        
        Returns:
            tuple: (grid, starts, goals) where starts and goals are lists of (x, y)
        """
        if seed is not None:
            random.seed(seed)
        
        grid = Grid(width, height, obstacle_density, seed)
        
        # Collect all free cells
        free_cells = [(x, y) for x in range(width) for y in range(height)
                      if (x, y) not in grid.obstacles]
        
        if len(free_cells) < 2 * num_agents:
            raise ValueError("Not enough free cells for all agents")
        
        # Try to find valid start/goal pairs
        max_attempts = 200
        for attempt in range(max_attempts):
            random.shuffle(free_cells)
            starts = free_cells[:num_agents]
            goals = free_cells[num_agents:2 * num_agents]
            
            # Verify all start-goal pairs are reachable
            all_reachable = True
            for s, g in zip(starts, goals):
                if not grid._path_exists(s, g):
                    all_reachable = False
                    break
            
            if all_reachable:
                grid.start = starts[0]
                grid.goal = goals[0]
                return grid, starts, goals
        
        # If we failed, reduce obstacles and retry
        grid = Grid(width, height, max(0.05, obstacle_density - 0.1), seed)
        free_cells = [(x, y) for x in range(width) for y in range(height)
                      if (x, y) not in grid.obstacles]
        random.shuffle(free_cells)
        starts = free_cells[:num_agents]
        goals = free_cells[num_agents:2 * num_agents]
        grid.start = starts[0]
        grid.goal = goals[0]
        return grid, starts, goals
    
    def __repr__(self):
        return f"Grid({self.width}x{self.height}, obstacles={len(self.obstacles)}, density={self.obstacle_density})"
