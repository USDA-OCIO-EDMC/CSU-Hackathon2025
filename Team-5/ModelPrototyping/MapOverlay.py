import pygame
from PIL import Image
import rasterio
import numpy as np
import sys
import pygame
import heapq


class Robot:
    def __init__(self, speed, width=20, height=20):
        self.robot = pygame.Rect(0, 0, width, height)
        self.speed = speed
        self.path = []
        self.route_points = []
        self.target = None
        self.current_segment = 0
        self.movement_type = "terrain"
        self.distance_threshold = 100
        self.terrain_data = None
        self.terrain_grid = None
        self.terrain_width = 0
        self.terrain_height = 0
        self.terrain_weight = 5.0
        self.max_elevation_diff = 20  # Maximum allowed elevation difference between adjacent cells
        self.elevation_weight = 10.0  # How strongly elevation differences affect path cost

    def load_terrain_data(self, filename):
        """Load terrain data from file and process it"""
        terrain_rows = []
        with open(filename, 'r') as file:
            for line in file:
                row = [int(val) for val in line.split()]
                terrain_rows.append(row)
        
        self.terrain_data = np.array(terrain_rows)
        self.terrain_height, self.terrain_width = self.terrain_data.shape
        print(f"Loaded terrain with dimensions: {self.terrain_width}x{self.terrain_height}")

    def find_path(self, target_pos):
        self.target = target_pos
        self.current_segment = 0
        
        if self.terrain_data is not None:
            self.movement_type = "terrain"
            self.find_terrain_path(target_pos)
        else:
            self.movement_type = "euclidean"
            self.update_route_points()

    def find_terrain_path(self, target_pos):
        """Find path considering both terrain values and elevation changes"""
        start_x, start_y = self.robot.centerx, self.robot.centery
        target_x, target_y = target_pos
        
        # Scale coordinates to match terrain grid
        start_grid_x = min(int(start_x * self.terrain_width / pygame.display.get_surface().get_width()), self.terrain_width - 1)
        start_grid_y = min(int(start_y * self.terrain_height / pygame.display.get_surface().get_height()), self.terrain_height - 1)
        target_grid_x = min(int(target_x * self.terrain_width / pygame.display.get_surface().get_width()), self.terrain_width - 1)
        target_grid_y = min(int(target_y * self.terrain_height / pygame.display.get_surface().get_height()), self.terrain_height - 1)
        
        path = self.elevation_aware_a_star((start_grid_y, start_grid_x), (target_grid_y, target_grid_x))
        
        if path:
            screen_width = pygame.display.get_surface().get_width()
            screen_height = pygame.display.get_surface().get_height()
            
            self.route_points = []
            self.route_points.append((self.robot.centerx, self.robot.centery))
            
            for grid_y, grid_x in path[1:]:
                screen_x = int(grid_x * screen_width / self.terrain_width)
                screen_y = int(grid_y * screen_height / self.terrain_height)
                self.route_points.append((screen_x, screen_y))
            
            self.path = self.route_points.copy()
            self.current_segment = 1
        else:
            self.movement_type = "euclidean"
            self.update_route_points()

    def elevation_aware_a_star(self, start, goal):
        """A* pathfinding that avoids steep elevation changes"""
        max_value = np.max(self.terrain_data)
        min_value = np.min(self.terrain_data)
        value_range = max_value - min_value
        
        open_set = []
        heapq.heappush(open_set, (0, 0, start, []))
        closed_set = set()
        
        while open_set:
            f, g, current, path = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            new_path = path + [current]
            
            if current == goal:
                return new_path
                
            closed_set.add(current)
            
            current_elev = self.terrain_data[current]
            
            for dy, dx in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                ny, nx = current[0] + dy, current[1] + dx
                
                if 0 <= ny < self.terrain_height and 0 <= nx < self.terrain_width:
                    if (ny, nx) not in closed_set:
                        neighbor_elev = self.terrain_data[ny, nx]
                        elev_diff = abs(neighbor_elev - current_elev)
                        
                        # Skip if elevation difference is too steep
                        if elev_diff > self.max_elevation_diff:
                            continue
                            
                        terrain_value = (self.terrain_data[ny, nx] - min_value) / value_range
                        terrain_cost = (1.0 - terrain_value) * self.terrain_weight
                        
                        # Add elevation difference cost
                        elev_cost = (elev_diff / self.max_elevation_diff) * self.elevation_weight
                        
                        if dx != 0 and dy != 0:  # Diagonal
                            move_cost = 1.414 + terrain_cost + elev_cost
                        else:  # Straight
                            move_cost = 1.0 + terrain_cost + elev_cost
                            
                        new_g = g + move_cost
                        
                        # Heuristic (Euclidean distance)
                        h = ((ny - goal[0]) ** 2 + (nx - goal[1]) ** 2) ** 0.5
                        f = new_g + h
                        
                        heapq.heappush(open_set, (f, new_g, (ny, nx), new_path))
        
        return None  # No path found within elevation constraints

    def update_route_points(self):
        if self.target:
            start_pos = (self.robot.centerx, self.robot.centery)
            if self.movement_type == "manhattan":
                self.route_points = [
                    start_pos,
                    (start_pos[0], self.target[1]),
                    self.target
                ]
                self.path = self.route_points.copy()
                self.current_segment = 1
            elif self.movement_type == "euclidean":
                self.route_points = [
                    start_pos,
                    self.target
                ]
                self.path = self.route_points.copy()
                self.current_segment = 1

    def update(self):
        if not self.target or not self.path:
            return

        try:
            if self.movement_type in ["terrain", "euclidean"]:
                if self.current_segment < len(self.path):
                    current_pos = (self.robot.centerx, self.robot.centery)
                    target_pos = self.path[self.current_segment]
                    
                    dx = target_pos[0] - current_pos[0]
                    dy = target_pos[1] - current_pos[1]
                    distance = (dx ** 2 + dy ** 2) ** 0.5
                    
                    if distance < self.speed:
                        self.robot.x = target_pos[0] - self.robot.width // 2
                        self.robot.y = target_pos[1] - self.robot.height // 2
                        self.current_segment += 1
                        
                        if self.current_segment >= len(self.path):
                            self.target = None
                            return
                    else:
                        move_x = (dx / distance) * self.speed
                        move_y = (dy / distance) * self.speed
                        self.robot.x += move_x
                        self.robot.y += move_y
            
            elif self.movement_type == "manhattan":
                if self.current_segment == 1:
                    current_pos = (self.robot.centerx, self.robot.centery)
                    target_y = self.path[1][1]
                    
                    if abs(current_pos[1] - target_y) > self.speed:
                        if current_pos[1] < target_y:
                            self.robot.y += self.speed
                        else:
                            self.robot.y -= self.speed
                    else:
                        self.robot.y = target_y - self.robot.height // 2
                        self.current_segment = 2
                
                elif self.current_segment == 2:
                    current_pos = (self.robot.centerx, self.robot.centery)
                    target_x = self.path[2][0]
                    
                    if abs(current_pos[0] - target_x) > self.speed:
                        if current_pos[0] < target_x:
                            self.robot.x += self.speed
                        else:
                            self.robot.x -= self.speed
                    else:
                        self.robot.x = target_x - self.robot.width // 2
                        self.target = None
                        return

        except (IndexError, ValueError):
            self.path = []
            self.route_points = []
            self.target = None

            
def load_raster_as_world(raster_file):
    """Loads a TIFF raster file as a Pygame surface."""
    try:
        img = Image.open(raster_file)
        
        # Convert to RGB or RGBA if necessary
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")
            
        world = pygame.image.fromstring(img.tobytes(), img.size, img.mode)
        return world
        
    except FileNotFoundError:
        print(f"Error: File '{raster_file}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

pygame.init()

# Load terrain data
with rasterio.open("shade.tiff") as src:
    data = src.read(1)
    with open("shade.txt", "w") as f:
        for row in data:
            f.write(" ".join(map(str, row)) + "\n")

# Create world surface
world = load_raster_as_world("shade.tiff")
if world is None:
    pygame.quit()
    sys.exit()

world_width, world_height = world.get_width(), world.get_height()
screen_width, screen_height = 800, 600  # Smaller window size for better navigation

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Robot Pathfinding with Terrain")
clock = pygame.time.Clock()

# Create robot and load terrain data
bot = Robot(2, width=10, height=10)
bot.load_terrain_data("shade.txt")

# Center robot in world
bot.robot.center = (world_width // 2, world_height // 2)

# Camera/view rectangle - starts centered on robot
camera = pygame.Rect(0, 0, screen_width, screen_height)
camera.center = bot.robot.center

# Add font for displaying current mode
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)

# Create a heatmap visualization overlay (optional)
def create_terrain_overlay(terrain_data, alpha=128):
    max_value = np.max(terrain_data)
    min_value = np.min(terrain_data)
    
    h, w = terrain_data.shape
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    
    for y in range(h):
        for x in range(w):
            value = terrain_data[y, x]
            normalized = int(255 * (value - min_value) / (max_value - min_value))
            overlay.set_at((x, y), (0, normalized, 0, alpha))
    
    return pygame.transform.scale(overlay, (world_width, world_height))

# For toggling overlay
show_overlay = False
show_help = True
panning = False
pan_start = (0, 0)
last_mouse_pos = (0, 0)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click - pan
                panning = True
                pan_start = event.pos
                last_mouse_pos = event.pos
            elif event.button == 3:  # Right click - set target
                # Convert screen to world coordinates
                world_x = event.pos[0] + camera.x
                world_y = event.pos[1] + camera.y
                bot.find_path((world_x, world_y))
                
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                panning = False
                
        elif event.type == pygame.MOUSEMOTION:
            if panning:
                # Get mouse movement since last frame
                dx = event.pos[0] - last_mouse_pos[0]
                dy = event.pos[1] - last_mouse_pos[1]
                
                # Move camera in opposite direction
                camera.x -= dx
                camera.y -= dy
                
                # Keep camera within world bounds
                camera.x = max(0, min(camera.x, world_width - camera.width))
                camera.y = max(0, min(camera.y, world_height - camera.height))
                
                last_mouse_pos = event.pos
                
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_o:
                show_overlay = not show_overlay
            elif event.key == pygame.K_h:
                show_help = not show_help
            elif event.key == pygame.K_m:
                # Cycle movement modes if implemented
                pass
            elif event.key == pygame.K_SPACE:
                # Center camera on robot
                camera.center = bot.robot.center
    
    # Update robot
    try:
        if bot.target:
            bot.update()
    except ValueError:
        bot.path = []
    
    # Clear screen
    screen.fill((0, 0, 0))
    
    # Draw visible portion of world
    screen.blit(world, (0, 0), camera)
    
    # Show terrain overlay if enabled
    if show_overlay and hasattr(bot, 'terrain_data') and bot.terrain_data is not None:
        overlay = create_terrain_overlay(bot.terrain_data, alpha=50)
        screen.blit(overlay, (0, 0), camera)
    
    # Draw route if there are route points (adjust coordinates for camera)
    if len(bot.route_points) > 1:
        adjusted_points = [(x - camera.x, y - camera.y) for (x, y) in bot.route_points]
        pygame.draw.lines(screen, (0, 255, 0), False, adjusted_points, 2)
        
        # Draw small circles at waypoints
        for point in adjusted_points:
            pygame.draw.circle(screen, (255, 255, 0), point, 3)
    
    # Draw the robot (adjust coordinates for camera)
    robot_rect = pygame.Rect(
        bot.robot.x - camera.x,
        bot.robot.y - camera.y,
        bot.robot.width,
        bot.robot.height
    )
    pygame.draw.rect(screen, (255, 0, 0), robot_rect)
    
    # Display current movement type
    text = font.render(f"Mode: {bot.movement_type}", True, (255, 255, 255))
    screen.blit(text, (10, 10))
    
    # Show help text
    if show_help:
        help_text = [
            "Right-click: Set robot target",
            "Left-click+drag: Pan camera",
            "Space: Center on robot",
            "O: Toggle terrain overlay",
            "H: Hide this help"
        ]
        
        for i, line in enumerate(help_text):
            help_line = small_font.render(line, True, (255, 255, 255))
            screen.blit(help_line, (10, screen_height - 120 + i*20))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()

