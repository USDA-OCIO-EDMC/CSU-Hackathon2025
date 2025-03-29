# robot_class.py
import pygame
import math
from algorithm import Algorithm

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.grid_x = x // 20  # Changed from 32 to 20
        self.grid_y = y // 20  # Changed from 32 to 20
        self.speed = 2
        self.robot = pygame.Rect(x - 10, y - 10, 20, 20)
        self.algorithm = Algorithm()
        self.current_path = []
        self.current_waypoint = 0
        self.has_path = False
        self.waypoints = []
        self.current_target_index = 0
        self.current_elevation = 0
        self.current_slope = 0

    def set_waypoints(self, points, grid):
        self.waypoints = points
        # Start from the robot's current grid position
        current_pos = (self.grid_x, self.grid_y)
        self.current_target_index = 0
        
        # Don't recalculate path if we're already at the first waypoint
        if current_pos != self.waypoints[0]:
            self.recalculate_path_from_current(grid)
        else:
            # If we're at the first waypoint, move to the second
            self.current_target_index = 1
            self.recalculate_path_from_current(grid)

    def recalculate_path_from_current(self, grid):
        if self.current_target_index < len(self.waypoints):
            current_pos = (self.grid_x, self.grid_y)
            target = self.waypoints[self.current_target_index]
            
            new_path = self.algorithm.find_path(current_pos, target, grid)
            
            if new_path:
                self.current_path = new_path
                self.current_waypoint = 0
                self.has_path = True
            else:
                self.has_path = False
        else:
            self.has_path = False

    def update(self, grid):
        if self.has_path and self.current_waypoint < len(self.current_path):
            # Get target position in grid coordinates
            target_grid = self.current_path[self.current_waypoint]
            # Convert to pixel coordinates (center of grid cell)
            target_x = target_grid[0] * 20 + 10  # Changed from 32 to 20
            target_y = target_grid[1] * 20 + 10  # Changed from 32 to 20

            dx = target_x - self.x
            dy = target_y - self.y
            distance = math.sqrt(dx * dx + dy * dy)


            if distance > self.speed:
                # Normalize direction and apply speed
                dx = (dx / distance) * self.speed
                dy = (dy / distance) * self.speed
                
                # Update position
                self.x += dx
                self.y += dy
                # Update grid position
                self.grid_x = int(self.x // 20)  # Changed from 32 to 20
                self.grid_y = int(self.y // 20)  # Changed from 32 to 20
                
                # Adjust robot size to be proportional to grid size
                robot_size = 10  # Half of old size to match smaller grid
                self.robot = pygame.Rect(self.x - robot_size, self.y - robot_size, robot_size*2, robot_size*2)

                # Update elevation
                if 0 <= self.grid_x < grid.shape[0] and 0 <= self.grid_y < grid.shape[1]:
                    self.current_elevation = grid[self.grid_x, self.grid_y, 1]
                    
                    # Calculate slope if we have a previous point
                    if self.current_waypoint > 0:
                        prev_grid = self.current_path[self.current_waypoint - 1]
                        self.current_slope = self.algorithm.get_slope(prev_grid, target_grid, grid)
            else:
                self.current_waypoint += 1
                if self.current_waypoint >= len(self.current_path):
                    self.current_target_index += 1
                    self.recalculate_path_from_current(grid)
        

    def draw(self, screen, grid, path_color=(255, 0, 0), robot_color=(0, 0, 255)):
        GRID_SIZE = 20  # Changed from 32 to 20
        
        # Draw path with brighter color and thicker lines
        if self.has_path and len(self.current_path) > 1:
            for i in range(len(self.current_path) - 1):
                start_pos = (self.current_path[i][0] * GRID_SIZE + GRID_SIZE//2,
                           self.current_path[i][1] * GRID_SIZE + GRID_SIZE//2)
                end_pos = (self.current_path[i+1][0] * GRID_SIZE + GRID_SIZE//2,
                          self.current_path[i+1][1] * GRID_SIZE + GRID_SIZE//2)
                pygame.draw.line(screen, path_color, start_pos, end_pos, 2)  # Slightly thinner line for smaller grid
                
                # Add dots at path points for better visibility
                pygame.draw.circle(screen, path_color, start_pos, 2)  # Smaller dots
            
            # Draw last point
            last_pos = (self.current_path[-1][0] * GRID_SIZE + GRID_SIZE//2,
                       self.current_path[-1][1] * GRID_SIZE + GRID_SIZE//2)
            pygame.draw.circle(screen, path_color, last_pos, 2)  # Smaller dot

        # Draw robot with customizable color
        pygame.draw.rect(screen, robot_color, self.robot)
        
        # Add a border around the robot for better visibility
        border = self.robot.copy()
        border.inflate_ip(2, 2)  # Smaller inflation for smaller robot
        pygame.draw.rect(screen, (0, 0, 0), border, 1)  # Thinner border
        
        # Draw debug information with white background for better visibility
        font = pygame.font.Font(None, 20)  # Smaller font
        
        # Create position text
        pos_text = font.render(f"Pos: ({int(self.x)}, {int(self.y)}) Grid: ({self.grid_x}, {self.grid_y})", True, (0, 0, 0))
        pos_rect = pos_text.get_rect(topleft=(10, 50))
        
        # Create background for position text
        bg_rect = pos_rect.copy()
        bg_rect.inflate_ip(8, 4)  # Smaller inflation for smaller text
        pygame.draw.rect(screen, (255, 255, 255), bg_rect)
        pygame.draw.rect(screen, (0, 0, 0), bg_rect, 1)
        
        # Display position text
        screen.blit(pos_text, pos_rect)

        # Display target information if we have a path
        if self.has_path and self.current_waypoint < len(self.current_path):
            target = self.current_path[self.current_waypoint]
            target_text = font.render(f"Target: {target}", True, (0, 0, 0))
            target_rect = target_text.get_rect(topleft=(10, 75))
            
            # Create background for target text
            bg_rect = target_rect.copy()
            bg_rect.inflate_ip(8, 4)  # Smaller inflation for smaller text
            pygame.draw.rect(screen, (255, 255, 255), bg_rect)
            pygame.draw.rect(screen, (0, 0, 0), bg_rect, 1)
            
            # Display target text
            screen.blit(target_text, target_rect)
            
    def draw_without_path(self, screen, grid, robot_color=(0, 0, 255)):
        """Draw only the robot without drawing the path"""
        # Draw robot with customizable color
        pygame.draw.rect(screen, robot_color, self.robot)
        
        # Add a border around the robot for better visibility
        border = self.robot.copy()
        border.inflate_ip(2, 2)  # Smaller inflation for smaller robot
        pygame.draw.rect(screen, (0, 0, 0), border, 1)  # Thinner border
        
        # Draw debug information with white background for better visibility
        font = pygame.font.Font(None, 20)  # Smaller font
        
        # Create position text
        pos_text = font.render(f"Pos: ({int(self.x)}, {int(self.y)}) Grid: ({self.grid_x}, {self.grid_y})", True, (0, 0, 0))
        pos_rect = pos_text.get_rect(topleft=(10, 50))
        
        # Create background for position text
        bg_rect = pos_rect.copy()
        bg_rect.inflate_ip(8, 4)  # Smaller inflation for smaller text
        pygame.draw.rect(screen, (255, 255, 255), bg_rect)
        pygame.draw.rect(screen, (0, 0, 0), bg_rect, 1)
        
        # Display position text
        screen.blit(pos_text, pos_rect)

    def handle_obstacle_change(self, grid):
        """Recalculate path when obstacles change"""
        self.recalculate_path_from_current(grid)