import math
import pygame

class Rover:
    def __init__(self, pos, road_network):
        self.geo_pos = pos  # Geographic position
        self.road_network = road_network
        self.path = []
        self.current_path_index = 0
        self.speed = 0.0001  # Speed in geographic units
        self.reached_destination = False
        self.destination = None

    def get_destination(self):
        return self.destination
        
    def set_destination(self, destination):
        self.destination = destination
        self.reached_destination = False
        return self.road_network.find_path(self.geo_pos, self.destination)
    
    def update(self):
        """Update rover position along the path"""
        if not self.path or self.current_path_index >= len(self.path) - 1:
            self.reached_destination = True
            return
        
        # Current and next waypoint
        current = self.path[self.current_path_index]
        next_wp = self.path[self.current_path_index + 1]
        
        # Direction vector
        dir_x = next_wp[0] - current[0]
        dir_y = next_wp[1] - current[1]
        
        # Normalize
        distance = math.sqrt(dir_x**2 + dir_y**2)
        if distance > 0:
            dir_x /= distance
            dir_y /= distance
        
        # Move rover
        self.geo_pos = (
            self.geo_pos[0] + dir_x * self.speed,
            self.geo_pos[1] + dir_y * self.speed
        )
        
        # Check if reached next waypoint
        new_distance = math.sqrt((next_wp[0] - self.geo_pos[0])**2 + 
                               (next_wp[1] - self.geo_pos[1])**2)
        if new_distance < self.speed:
            self.current_path_index += 1
            
            # Check if reached final destination
            if self.current_path_index >= len(self.path) - 1:
                self.reached_destination = True
    
    def draw(self, screen):
        """Draw the rover on the Pygame screen"""
        screen_x, screen_y = self.road_network.geo_to_screen(self.geo_pos[0], self.geo_pos[1])
        
        # Draw rover as a blue circle
        pygame.draw.circle(screen, (0, 0, 255), (screen_x, screen_y), 5)
        
        # Add a direction indicator (optional)
        if not self.reached_destination and len(self.path) > 1 and self.current_path_index < len(self.path) - 1:
            next_wp = self.path[self.current_path_index + 1]
            next_x, next_y = self.road_network.geo_to_screen(next_wp[0], next_wp[1])
            pygame.draw.line(screen, (0, 255, 255), (screen_x, screen_y), (next_x, next_y), 2)