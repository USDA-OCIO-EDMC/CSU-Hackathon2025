import pygame
import sys
from road_networks import RoadNetwork
from robot_class import Rover

def main():
    # Initialize Pygame
    pygame.init()
    screen_width, screen_height = 1000, 1000
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Road Network A* Navigation")
    
    # Load road network from shapefile
    # Replace this with your actual shapefile path

    shapefile_path = "South_Clear_Creek_Roads.shp"
    road_network = RoadNetwork(shapefile_path, screen_width, screen_height)
    
    # Font for displaying information
    font = pygame.font.SysFont(None, 24)
    
    # Data structure to store vector data
    vector_data = []
    rover_initialized = False
    rover = None
    path = []

    # Main game loop
    running = True
    clock = pygame.time.Clock()
    show_instructions = True
    show_stats = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Set new destination when clicking on screen
                mouse_x, mouse_y = pygame.mouse.get_pos()
                print(f"Mouse clicked at: ({mouse_x}, {mouse_y})")

                # Convert screen coordinates to geographic coordinates
                click_pos = road_network.screen_to_geo(mouse_x, mouse_y)
                
                # Find the nearest road point
                _, nearest_pos = road_network.find_nearest_node(click_pos)
                print(f"Nearest road point: {nearest_pos}")

                new_pos = nearest_pos
                vector_data.append(new_pos)
                print(f"Point set at: {new_pos}")

                if not rover_initialized:
                    rover = Rover(new_pos, road_network)
                    rover_initialized = True
                    print("Rover initialized.")
                elif rover_initialized:
                    path = rover.set_destination(new_pos)
                    print(f"Destination set: {new_pos}")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    # Toggle instructions
                    show_instructions = not show_instructions
                elif event.key == pygame.K_s:
                    # Toggle statistics
                    show_stats = not show_stats
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        # Update rover
        #if rover is not None:
        #    rover.update()
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw everything
        road_network.draw(screen)
        if rover is not None:
            road_network.draw_path(screen, path)
            rover.draw(screen)
            road_network.draw_destination(screen, rover.get_destination())
        
        # Display path statistics if enabled
        if show_stats:
            road_network.draw_path_stats(screen, path, font)
        
        # Display instructions
        if show_instructions:
            lines = [
                "Click anywhere to set a destination",
                "Press H to toggle instructions",
                "Press S to toggle statistics",
                "Press ESC to exit",
                "Using A* algorithm for optimal path finding"
            ]
            
            for i, line in enumerate(lines):
                text = font.render(line, True, (255, 255, 255))
                screen.blit(text, (10, 10 + i * 25))
        
        # Display destination status
        if rover is not None and rover.reached_destination:
            text = font.render("Destination reached!", True, (0, 255, 0))
            screen.blit(text, (screen_width - 200, 10))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()