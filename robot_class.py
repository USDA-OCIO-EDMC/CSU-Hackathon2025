import math

import pygame

from node import Node


def dist(pos1, pos2):
    """
    Calculate the Euclidean distance between two points.
    """
    return math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)

def reconstruct_path(came_from, current):
    total_path = [current]
    for current in came_from.keys():
        total_path.insert(0, came_from[current])
    return total_path

class Robot:
    def __init__(self, speed, start : Node, end : Node, nodes : list):
        self.nodes = nodes
        self.speed = speed
        self.start = start
        self.end = end
        self.robot = pygame.Rect(start.x, start.y, 10, 10)
        self.count = 0
        self.open_lis = [self.start]
        self.closed_lis = []
        self.came_from = {}
        self.g_score = {node: float('inf') for node in nodes}  # Initialize g_score for all nodes
        self.f_score = {node: float('inf') for node in nodes}  # Initialize f_score for all nodes
        self.g_score[self.start] = 0
        self.f_score[self.start] = dist(self.start.p, self.end.p)
        self.path = self.a_star()


    def a_star(self):
        print("open_lis: ", self.open_lis)
        while len(self.open_lis) > 0:
            current = min(self.open_lis, key=lambda node: self.f_score[node])
            print("current found with f_score: ", self.f_score[current])
            if current == self.end:
                print("End reached, reconstructing path:")
                return reconstruct_path(self.came_from, current)

            self.open_lis.remove(current)
            print("debugging", current.neighbors, current)
            for neighbor in current.neighbors:
                # if neighbor in self.closed_lis:
                #     current.neighbors.remove(neighbor)
                #     continue
                self.closed_lis.append(neighbor)

                tentative_g_score = self.g_score[current] + dist(current.p, neighbor.p)
                if tentative_g_score < self.g_score[neighbor]:
                    print("new neighbor")
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    self.f_score[neighbor] = tentative_g_score + dist(neighbor, self.end.p)
                    if neighbor not in self.open_lis:
                        self.open_lis.append(neighbor)
        print("No path found")
        return None


    def update(self):
        if self.path:
            self.move_towards(self.path[self.count])
            self.count += 1




    def move_towards(self, node: Node):
        """
        move from current pos to given node.pos, moving through all the nodes in the sequence on the way
        """
        target = node.p
        dx = target[0] - self.robot.centerx
        dy = target[1] - self.robot.centery
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist < self.speed or node == self.end:
            return
        dirx = dx / dist
        diry = dy / dist
        self.robot.move_ip(dirx * self.speed, diry * self.speed)
        print(f"Moving towards: {target}, Direction X: {dirx}, Direction Y: {diry}")


    def move_left(self):
        self.robot.move_ip(-self.speed, 0)

    def move_right(self):
        self.robot.move_ip(self.speed, 0)

    def move_up(self):
        self.robot.move_ip(0, -self.speed)

    def move_down(self):
        self.robot.move_ip(0, self.speed)
