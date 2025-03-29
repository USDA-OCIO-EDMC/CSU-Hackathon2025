import matplotlib.pyplot as plt

from Visualizer import visualize
from A import Algorithm


class Robot:
    def __init__(self, speed, grid, src, dest):
        self.speed = speed
        self.robot = plt.Circle((src[1], src[0]), 10, color='red')
        self.grid = grid
        self.src = src
        self.dest = dest
        self.path = Algorithm.a_start_search(grid, src, dest)
        self.current_target_index = 0

    def update(self):
        if self.path:
            if self.current_target_index < len(self.path):
                target = self.path[self.current_target_index]
                if self.robot.centerx < target[1]:
                    self.move_right()
                elif self.robot.centerx > target[1]:
                    self.move_left()
                if self.robot.centery < target[0]:
                    self.move_down()
                elif self.robot.centery > target[0]:
                    self.move_up()
                centerx_diff = abs(self.robot.centerx - target[1])
                centery_diff = abs(self.robot.centery - target[0])
                if centerx_diff < self.speed and centery_diff < self.speed:
                    self.current_target_index += 1

                self.visualize_path()

    def move_left(self):
        self.robot.move_ip(-self.speed, 0)

    def move_right(self):
        self.robot.move_ip(self.speed, 0)

    def move_up(self):
        self.robot.move_ip(0, -self.speed)

    def move_down(self):
        self.robot.move_ip(0, self.speed)

    def visualize_path(self):
        # Get current position (robot's center) and path to visualize
        current_position = (self.robot.center[1], self.robot.center[0])

        # Visualize the path after each update
        visualize.plotVisual(self.path)
        plt.plot(current_position[0], current_position[1], 'bo')
        plt.pause(0.01)