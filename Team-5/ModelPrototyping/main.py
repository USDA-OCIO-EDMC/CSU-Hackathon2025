# main.py
import pygame
import sys
import numpy as np
from robot_class import Robot
import os
import random
from algorithm import Algorithm


# main.py
def main():
    pygame.init()
    
    # Updated Grid settings for smaller cells but more spaces
    GRID_SIZE = 20  # Smaller grid cells (changed from 32)
    GRID_WIDTH = 40  # More horizontal spaces
    GRID_HEIGHT = 30  # More vertical spaces

    WINDOW_SIZE = (GRID_WIDTH * GRID_SIZE, GRID_HEIGHT * GRID_SIZE)  # 800x600 window
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Robot Pathfinding with Elevation")
    
    # Keep your existing color definitions...
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (180, 180, 180)  # Lighter gray for obstacles
    ROBOT_COLOR = (0, 100, 255)  # Brighter blue for robot
    PATH_COLOR = (255, 50, 50)   # Brighter red for path
    WAYPOINT_COLOR = (0, 180, 255)  # Bright blue for waypoints
    GRID_LINE_COLOR = (100, 100, 100)  # Lighter grid lines
    TEXT_COLOR = (0, 0, 0)  # Black text
    STEEP_SLOPE_COLOR = (255, 50, 50, 128)  # Semi-transparent red for steep slopes
    CLIFF_COLOR = (128, 0, 0, 180)  # Darker, more transparent red for cliffs
    TREE_COLOR = (20, 120, 20)  # Dark green for trees
    ROCK_COLOR = (150, 150, 150)  # Gray for rocks
    # Create grid with more spaces
    grid = np.zeros((GRID_WIDTH, GRID_HEIGHT, 3))

    # Adjust terrain generation for more detailed landscape
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            # Adjusted scale for more varied terrain
            elevation = (np.sin(x/4) * np.cos(y/4) * 2) + 2
            elevation += random.uniform(-0.2, 0.2)
            
            # Adjust valley location and size
            center_x, center_y = GRID_WIDTH//3, GRID_HEIGHT//2
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist < 8:  # Larger valley
                elevation -= (8 - dist) * 0.3
                
            # Adjust hill location and size
            hill_x, hill_y = 2*GRID_WIDTH//3, GRID_HEIGHT//3
            dist = np.sqrt((x - hill_x)**2 + (y - hill_y)**2)
            if dist < 10:  # Larger hill
                elevation += (10 - dist) * 0.3
                
            grid[x, y, 1] = elevation

    # Adjust wall positions for new grid size
    for y in range(8, 18):
        if y != 13:  # Leave a gap
            grid[15, y, 0] = 1
            grid[15, y, 2] = 1

    for x in range(8, 23):
        if x != 18:  # Leave a gap
            grid[x, 18, 0] = 1
            grid[x, 18, 2] = 1

    # Adjust number of trees and rocks for larger grid
    num_trees = 20  # More trees
    num_rocks = 12  # More rocks
    
    object_positions = []
    MAX_SLOPE = 45
    # Place trees
    for _ in range(num_trees):
        while True:
            x = random.randint(1, GRID_WIDTH-2)
            y = random.randint(1, GRID_HEIGHT-2)
            if grid[x, y, 0] == 0:
                grid[x, y, 0] = 1
                grid[x, y, 2] = 2
                object_positions.append((x, y))
                break
    
    # Place rocks
    for _ in range(num_rocks):
        while True:
            x = random.randint(1, GRID_WIDTH-2)
            y = random.randint(1, GRID_HEIGHT-2)
            if grid[x, y, 0] == 0:
                grid[x, y, 0] = 1
                grid[x, y, 2] = 3
                object_positions.append((x, y))
                break

    # Update tree drawing function for smaller grid size
    def draw_tree(screen, x, y):
        trunk_color = (101, 67, 33)
        leaves_color = (0, 128, 0)
        
        # Adjusted for 20x20 grid
        trunk_rect = pygame.Rect(x * GRID_SIZE + 8, y * GRID_SIZE + 10, 4, 10)
        pygame.draw.rect(screen, trunk_color, trunk_rect)
        
        leaves_points = [
            (x * GRID_SIZE + 10, y * GRID_SIZE + 3),  # Top
            (x * GRID_SIZE + 3, y * GRID_SIZE + 12),  # Bottom left
            (x * GRID_SIZE + 17, y * GRID_SIZE + 12)  # Bottom right
        ]
        pygame.draw.polygon(screen, leaves_color, leaves_points)
    
    # Update rock drawing function for smaller grid size
    def draw_rock(screen, x, y):
        rock_color = (130, 130, 130)
        
        center_x = x * GRID_SIZE + GRID_SIZE//2
        center_y = y * GRID_SIZE + GRID_SIZE//2
        radius = GRID_SIZE//2 - 2  # Slightly smaller radius
        
        points = []
        num_points = 8
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            r = radius * (0.8 + random.random() * 0.4)
            px = center_x + r * np.cos(angle)
            py = center_y + r * np.sin(angle)
            points.append((px, py))
        
        pygame.draw.polygon(screen, rock_color, points)
        pygame.draw.line(screen, (160, 160, 160), 
                        (center_x - 3, center_y - 3), 
                        (center_x + 1, center_y - 1), 1)

    path_algorithm = Algorithm()
    impassable_cells = path_algorithm.calculate_impassable_terrain(grid)

    # Initialize robot with adjusted grid size
    while True:
        start_grid_x = random.randint(2, GRID_WIDTH//4)
        start_grid_y = random.randint(2, GRID_HEIGHT//4)
        if grid[start_grid_x, start_grid_y, 0] == 0 and (start_grid_x, start_grid_y) not in impassable_cells:
            break
            
    # Convert grid coordinates to pixel coordinates (center of grid cell)
    start_x = start_grid_x * GRID_SIZE + GRID_SIZE // 2
    start_y = start_grid_y * GRID_SIZE + GRID_SIZE // 2
    robot = Robot(start_x, start_y)
    
    # Choose destination with adjusted grid size
    while True:
        end_grid_x = random.randint(3*GRID_WIDTH//4, GRID_WIDTH-2)
        end_grid_y = random.randint(3*GRID_HEIGHT//4, GRID_HEIGHT-2)
        if grid[end_grid_x, end_grid_y, 0] == 0 and (end_grid_x, end_grid_y) not in impassable_cells:
            break
    
    waypoints = [(start_grid_x, start_grid_y), (end_grid_x, end_grid_y)]
    robot.set_waypoints(waypoints, grid)
    
    def get_elevation_color(elevation):
        """Convert elevation to a valid RGB color - using a brighter color scheme"""
        normalized = max(0, min(1, elevation / 4.0))  # Adjust scale for better visibility
        
        # Use a green-blue gradient that's easier to see
        r = int(150 + 105 * (1 - normalized))  # Higher elevation is lighter
        g = int(180 + 75 * normalized)
        b = int(100 + 155 * normalized)
        
        return (r, g, b)

    # Load tree and rock images or use simple shapes
    def draw_tree(screen, x, y):
        # Draw a simple tree using shapes
        trunk_color = (101, 67, 33)  # Brown
        leaves_color = (0, 128, 0)   # Green
        
        # Trunk
        trunk_rect = pygame.Rect(x * GRID_SIZE + 12, y * GRID_SIZE + 15, 8, 17)
        pygame.draw.rect(screen, trunk_color, trunk_rect)
        
        # Leaves (triangular top)
        leaves_points = [
            (x * GRID_SIZE + 16, y * GRID_SIZE + 5),  # Top
            (x * GRID_SIZE + 4, y * GRID_SIZE + 18),  # Bottom left
            (x * GRID_SIZE + 28, y * GRID_SIZE + 18)  # Bottom right
        ]
        pygame.draw.polygon(screen, leaves_color, leaves_points)
    
    def draw_rock(screen, x, y):
        # Draw a simple rock using an irregular shape
        rock_color = (130, 130, 130)  # Gray
        
        center_x = x * GRID_SIZE + 16
        center_y = y * GRID_SIZE + 16
        radius = 14
        
        # Create irregular rock shape
        points = []
        num_points = 8
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            r = radius * (0.8 + random.random() * 0.4)  # Vary radius for irregularity
            px = center_x + r * np.cos(angle)
            py = center_y + r * np.sin(angle)
            points.append((px, py))
        
        pygame.draw.polygon(screen, rock_color, points)
        # Add some highlights
        pygame.draw.line(screen, (160, 160, 160), 
                         (center_x - 5, center_y - 5), 
                         (center_x + 2, center_y - 2), 2)
    clock = pygame.time.Clock()
    running = True
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    # Track if slope visualization is enabled
    show_slopes = True
    show_cliffs = True
    
    # Track if the path is shown
    show_path = True
    
    # Create an elevation profile display area
    elevation_display = {
        'x': 50,
        'y': WINDOW_SIZE[1] - 150,
        'width': WINDOW_SIZE[0] - 100,
        'height': 100
    }

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_pos = pygame.mouse.get_pos()
                    grid_x = mouse_pos[0] // GRID_SIZE
                    grid_y = mouse_pos[1] // GRID_SIZE
                    if (grid_x, grid_y) not in waypoints:
                        grid[grid_x, grid_y, 0] = 1 - grid[grid_x, grid_y, 0]
                        # If removing obstacle, clear the obstacle type
                        if grid[grid_x, grid_y, 0] == 0:
                            grid[grid_x, grid_y, 2] = 0
                        else:
                            # If adding obstacle, make it a wall
                            grid[grid_x, grid_y, 2] = 1
                        
                        # Recalculate impassable terrain
                        impassable_cells = path_algorithm.calculate_impassable_terrain(grid)
                        
                        # Update robot path
                        robot.handle_obstacle_change(grid)
                        
                elif event.button == 3:  # Right click
                    # Toggle slope visualization
                    show_slopes = not show_slopes
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    # Toggle cliff visualization
                    show_cliffs = not show_cliffs
                elif event.key == pygame.K_p:
                    # Toggle path visualization
                    show_path = not show_path
                elif event.key == pygame.K_r:
                    # Generate new random destination
                    while True:
                        end_grid_x = random.randint(GRID_WIDTH//2, GRID_WIDTH-2)
                        end_grid_y = random.randint(GRID_HEIGHT//2, GRID_HEIGHT-2)
                        if grid[end_grid_x, end_grid_y, 0] == 0 and (end_grid_x, end_grid_y) not in impassable_cells:
                            break
                    waypoints[1] = (end_grid_x, end_grid_y)
                    robot.set_waypoints(waypoints, grid)

        # Update robot position
        robot.update(grid)

        # Clear screen
        screen.fill(WHITE)
        
        # Draw grid and terrain
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                # Draw terrain base color based on elevation
                color = get_elevation_color(grid[x, y, 1])
                pygame.draw.rect(screen, color, rect)
                
                # Draw grid lines - thinner and less intrusive
                pygame.draw.rect(screen, GRID_LINE_COLOR, rect, 1)
        
        # Draw impassable cells (steep slopes) - directly from algorithm
        if show_slopes:
            for x, y in impassable_cells:
                slope_rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                slope_surface = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(slope_surface, STEEP_SLOPE_COLOR, slope_surface.get_rect())
                screen.blit(slope_surface, slope_rect)
        
        # Draw cliff warnings
        if show_cliffs:
            for x, y in impassable_cells:
                # Check if this is specifically a cliff (steep downward slope)
                is_cliff = False
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]):
                        elev1 = grid[x, y, 1]
                        elev2 = grid[nx, ny, 1]
                        if elev1 - elev2 > 1.0:  # Significant drop
                            is_cliff = True
                            break
                
                if is_cliff:
                    cliff_rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                    cliff_surface = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
                    pygame.draw.rect(cliff_surface, CLIFF_COLOR, cliff_surface.get_rect())
                    
                    # Draw a warning symbol for cliffs
                    danger_color = (255, 0, 0)
                    center_x = x * GRID_SIZE + GRID_SIZE // 2
                    center_y = y * GRID_SIZE + GRID_SIZE // 2
                    
                    # Draw warning triangle
                    triangle_points = [
                        (center_x, center_y - 10),
                        (center_x - 8, center_y + 5),
                        (center_x + 8, center_y + 5)
                    ]
                    pygame.draw.polygon(screen, danger_color, triangle_points)
                    pygame.draw.polygon(screen, BLACK, triangle_points, 2)
                    
                    # Draw exclamation mark
                    pygame.draw.line(screen, BLACK, (center_x, center_y - 5), (center_x, center_y + 1), 2)
                    pygame.draw.line(screen, BLACK, (center_x, center_y + 3), (center_x, center_y + 3), 2)
                    
                    screen.blit(cliff_surface, cliff_rect)
                    
        # Draw obstacles on top of terrain
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if grid[x, y, 0] == 1:
                    obstacle_type = grid[x, y, 2]
                    if obstacle_type == 1:  # Wall
                        rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                        pygame.draw.rect(screen, GRAY, rect)
                    elif obstacle_type == 2:  # Tree
                        draw_tree(screen, x, y)
                    elif obstacle_type == 3:  # Rock
                        draw_rock(screen, x, y)

        # Draw robot and its path
        if show_path:
            robot.draw(screen, grid, PATH_COLOR, ROBOT_COLOR)
        else:
            # Draw robot without path
            robot.draw_without_path(screen, grid, ROBOT_COLOR)

        # Draw waypoint markers
        for i, point in enumerate(waypoints):
            # Convert grid coordinates to pixel coordinates
            pixel_x = point[0] * GRID_SIZE + GRID_SIZE // 2
            pixel_y = point[1] * GRID_SIZE + GRID_SIZE // 2
            
            # Draw waypoint marker - larger and more visible
            pygame.draw.circle(screen, WAYPOINT_COLOR, (pixel_x, pixel_y), 8)
            
            # Draw waypoint label with background for better visibility
            label = 'A' if i == 0 else 'B'
            text = font.render(label, True, TEXT_COLOR)
            text_rect = text.get_rect(center=(pixel_x, pixel_y - 20))
            
            # Add white background behind text for better visibility
            bg_rect = text_rect.copy()
            bg_rect.inflate_ip(10, 6)
            pygame.draw.rect(screen, WHITE, bg_rect)
            pygame.draw.rect(screen, BLACK, bg_rect, 1)
            
            screen.blit(text, text_rect)

        # Draw elevation profile if robot has a path
        if path_algorithm.elevation_profile and show_path:
            path_algorithm.draw_elevation_profile(
                screen, 
                elevation_display['x'], 
                elevation_display['y'], 
                elevation_display['width'], 
                elevation_display['height'],
                grid
            )

        # Draw instructions with background for better visibility
        info_text = font.render("Left: Add/Remove Wall | Right: Toggle Slopes | C: Toggle Cliffs | P: Toggle Path | R: New Destination", True, TEXT_COLOR)
        info_rect = info_text.get_rect(topleft=(10, 10))
        bg_rect = info_rect.copy()
        bg_rect.inflate_ip(10, 6)
        pygame.draw.rect(screen, WHITE, bg_rect)
        pygame.draw.rect(screen, BLACK, bg_rect, 1)
        screen.blit(info_text, info_rect)

        # Draw current elevation and slope info with background
        elevation_text = font.render(f"Elevation: {robot.current_elevation:.1f}", True, TEXT_COLOR)
        elevation_rect = elevation_text.get_rect(topleft=(10, 50))
        bg_rect_1 = elevation_rect.copy()
        bg_rect_1.inflate_ip(10, 6)
        pygame.draw.rect(screen, WHITE, bg_rect_1)
        pygame.draw.rect(screen, BLACK, bg_rect_1, 1)
        screen.blit(elevation_text, elevation_rect)
        
        slope_text = font.render(f"Slope: {robot.current_slope:.1f}° | Max: {MAX_SLOPE}°", True, TEXT_COLOR)
        slope_rect = slope_text.get_rect(topleft=(10, 80))
        bg_rect_2 = slope_rect.copy()
        bg_rect_2.inflate_ip(10, 6)
        pygame.draw.rect(screen, WHITE, bg_rect_2)
        pygame.draw.rect(screen, BLACK, bg_rect_2, 1)
        screen.blit(slope_text, slope_rect)
        
        # Draw status info
        status_texts = []
        if robot.has_path:
            status_texts.append("Status: Moving to destination")
        else:
            status_texts.append("Status: No path found or reached destination")
        
        status_texts.append(f"Show Slopes: {'On' if show_slopes else 'Off'}")
        status_texts.append(f"Show Cliffs: {'On' if show_cliffs else 'Off'}")
        status_texts.append(f"Show Path: {'On' if show_path else 'Off'}")
        
        for i, text in enumerate(status_texts):
            status_text = small_font.render(text, True, TEXT_COLOR)
            status_rect = status_text.get_rect(topleft=(10, 110 + i * 25))
            bg_rect = status_rect.copy()
            bg_rect.inflate_ip(10, 6)
            pygame.draw.rect(screen, WHITE, bg_rect)
            pygame.draw.rect(screen, BLACK, bg_rect, 1)
            screen.blit(status_text, status_rect)

        # Update display
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()