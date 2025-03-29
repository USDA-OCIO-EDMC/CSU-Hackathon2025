# Modified algorithm.py
import numpy as np
import heapq
import pygame
from bresenham import bresenham

class Algorithm:
    def __init__(self):
        self.elevation_profile = []
        self.max_slope = 45  # Maximum slope in degrees
        self.cliff_threshold = 1.0  # Minimum elevation drop to be considered a cliff
        # Store all impassable cells (too steep) for visualization
        self.impassable_cells = set()
        
    def calculate_impassable_terrain(self, grid):
        """Pre-calculate all cells that are impassable due to steep slopes"""
        self.impassable_cells.clear()
        
        # Loop through all grid cells
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                # Skip cells that are already obstacles
                if grid[x, y, 0] == 1:
                    continue
                    
                # Check all neighbors
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    # Check if neighbor is in bounds
                    if (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]):
                        # Calculate slope
                        elev1 = grid[x, y, 1]
                        elev2 = grid[nx, ny, 1]
                        elevation_diff = abs(elev2 - elev1)
                        distance = 1.0
                        slope = np.degrees(np.arctan2(elevation_diff, distance))
                        
                        # If slope is too steep in any direction, mark cell as impassable
                        if slope > self.max_slope:
                            self.impassable_cells.add((x, y))
                            break
        
        return self.impassable_cells

    def get_neighbors(self, pos, grid):
        neighbors = []
        # Only cardinal directions (remove diagonals for simpler movement)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for dx, dy in directions:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            
            # Check bounds
            if (0 <= new_x < grid.shape[0] and 
                0 <= new_y < grid.shape[1]):
                
                # Skip if it's an obstacle or in impassable cells list
                if grid[new_x, new_y, 0] == 1 or (new_x, new_y) in self.impassable_cells:
                    continue
                
                # Check slope between current and new cell
                current_elevation = grid[pos[0], pos[1], 1]
                new_elevation = grid[new_x, new_y, 1]
                
                # Calculate elevation difference
                elevation_diff = new_elevation - current_elevation
                
                # Calculate slope in degrees
                distance = 1.0  # Since we're only using cardinal directions
                slope = np.degrees(np.arctan2(abs(elevation_diff), distance))
                
                # Check if slope is navigable (not too steep)
                if slope <= self.max_slope:
                    # Extra check for downward cliffs
                    if elevation_diff < 0 and abs(elevation_diff) > self.cliff_threshold:
                        # This is a significant drop - check if it's too steep to go down
                        downward_slope = np.degrees(np.arctan2(abs(elevation_diff), distance))
                        if downward_slope <= self.max_slope:
                            neighbors.append((new_x, new_y))
                    else:
                        # Normal traversable terrain
                        neighbors.append((new_x, new_y))
        
        return neighbors

    def get_slope(self, pos1, pos2, grid):
        """Calculate slope between two positions"""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance == 0:
            return 0
            
        elevation_diff = grid[pos2[0], pos2[1], 1] - grid[pos1[0], pos1[1], 1]
        return np.degrees(np.arctan2(abs(elevation_diff), distance))

    def heuristic(self, a, b, grid=None):
        """
        A* heuristic with hybrid distance calculation.
        Uses Euclidean by default, switches to Manhattan when navigating around obstacles.
        
        Parameters:
        a: Current position (x, y)
        b: Goal position (x, y)
        grid: The terrain and obstacle grid
        """
        # Calculate direct Euclidean distance as baseline
        euclidean_dist = np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
        
        if grid is None:
            return euclidean_dist
        
        # Check if there's a clear line of sight between current position and goal
        if self.has_line_of_sight(a, b, grid):
            # Use Euclidean distance when no obstacles in between
            return euclidean_dist
        else:
            # Use Manhattan distance when navigating around obstacles
            # Apply a small weight to favor paths that make progress toward goal
            manhattan_dist = abs(b[0] - a[0]) + abs(b[1] - a[1])
            return manhattan_dist * 1.1
    
    def has_line_of_sight(self, start, end, grid):
        """
        Check if there's a clear line of sight between start and end positions.
        Uses Bresenham's line algorithm to check cells between points.
        
        Returns:
        True if there's a clear path, False if obstacles or impassable terrain in the way
        """
        # Get all cells on the line using Bresenham's algorithm
        line_cells = list(bresenham(start[0], start[1], end[0], end[1]))
        
        # Check each cell on the line
        prev_cell = start
        for cell in line_cells[1:]:  # Skip the starting cell
            x, y = cell
            
            # Check boundaries
            if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
                return False
                
            # Check if cell is an obstacle or impassable
            if grid[x, y, 0] == 1 or (x, y) in self.impassable_cells:
                return False
                
            # Check if slope between cells is acceptable
            if prev_cell != cell:  # Skip checking slope for the same cell
                slope = self.get_slope(prev_cell, cell, grid)
                if slope > self.max_slope:
                    return False
                    
            prev_cell = cell
            
        return True

    def find_direct_path(self, start, goal, grid):
        """
        Try to find a direct path between start and goal without A*.
        Returns the path if possible, None otherwise.
        """
        if self.has_line_of_sight(start, goal, grid):
            # Get path cells using Bresenham's algorithm
            line_cells = list(bresenham(start[0], start[1], goal[0], goal[1]))
            
            # Store elevation profile for visualization
            self.elevation_profile = [(pos, grid[pos[0], pos[1], 1]) for pos in line_cells]
            
            return line_cells
        
        return None

    def find_path(self, start, goal, grid):
        """Find a path using A* algorithm, avoiding steep terrain"""
        # First, calculate impassable terrain based on slopes
        self.calculate_impassable_terrain(grid)
        
        # If start or goal is in impassable terrain, return None
        if start in self.impassable_cells or goal in self.impassable_cells:
            print(f"Start {start} or goal {goal} is in impassable terrain!")
            return None
            
        # NEW: Try direct path first if possible
        direct_path = self.find_direct_path(start, goal, grid)
        if direct_path:
            print("Direct path found!")
            return direct_path
            
        print("Direct path not possible, using A* algorithm...")
        
        # Standard A* implementation
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break
                
            for next_pos in self.get_neighbors(current, grid):
                # Calculate new cost including elevation difference
                elevation_diff = abs(grid[next_pos[0], next_pos[1], 1] - 
                                  grid[current[0], current[1], 1])
                
                # Higher penalty for going uphill vs downhill
                if grid[next_pos[0], next_pos[1], 1] > grid[current[0], current[1], 1]:
                    # Going uphill - higher cost
                    movement_cost = 1 + elevation_diff * 0.8
                else:
                    # Going downhill - lower cost but still some penalty
                    movement_cost = 1 + elevation_diff * 0.3
                
                new_cost = cost_so_far[current] + movement_cost
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(next_pos, goal, grid)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        # Reconstruct path
        if goal not in came_from:
            print(f"No path found from {start} to {goal}!")
            return None
            
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        
        # Store elevation profile for visualization
        self.elevation_profile = [(pos, grid[pos[0], pos[1], 1]) for pos in path]
        
        return path

    def draw_elevation_profile(self, screen, x, y, width, height, grid):
        """Visualize the elevation profile of the current path"""
        if not self.elevation_profile:
            return
            
        # Draw background
        pygame.draw.rect(screen, (255, 255, 255), (x, y, width, height))
        pygame.draw.rect(screen, (0, 0, 0), (x, y, width, height), 1)
        
        # Get elevation range
        elevations = [e for _, e in self.elevation_profile]
        min_elevation = min(elevations)
        max_elevation = max(elevations)
        elevation_range = max_elevation - min_elevation or 1
        
        # Draw profile
        points = []
        for i, (_, elevation) in enumerate(self.elevation_profile):
            px = x + (i / (len(self.elevation_profile)-1)) * width
            py = y + height - ((elevation - min_elevation) / elevation_range) * height
            points.append((int(px), int(py)))
            
        if len(points) > 1:
            pygame.draw.lines(screen, (0, 0, 255), False, points, 2)
            
            # Mark steep sections
            for i in range(len(points)-1):
                pos1 = self.elevation_profile[i][0]
                pos2 = self.elevation_profile[i+1][0]
                
                # Calculate slope
                slope = self.get_slope(pos1, pos2, grid)
                
                if slope > self.max_slope * 0.8:  # Warning for slopes near maximum
                    # Draw a warning marker
                    mid_x = (points[i][0] + points[i+1][0]) // 2
                    mid_y = (points[i][1] + points[i+1][1]) // 2
                    pygame.draw.circle(screen, (255, 0, 0), (mid_x, mid_y), 4)