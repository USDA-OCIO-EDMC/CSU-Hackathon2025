from itertools import combinations

import pygame
import geopandas as gpd
from pygame.draw_py import draw_line
import csv

from node import Node
from edge import Edge
from robot_class import Robot

pygame.init()
clock = pygame.time.Clock()


h = 900
w = (10460 / 13115) * h
screen = pygame.display.set_mode((w, h))



extrema = {'minX' : 10000000,
               'maxX' : 0,
               'minY': 10000000,
               'maxY': 0
            }
extrema2 = {'minX' : 10000000,
               'maxX' : 0,
               'minY': 10000000,
               'maxY': 0
            }
extrema3 = {'minX' : 10000000,
               'maxX' : 0,
               'minY': 10000000,
               'maxY': 0
            }

def findExtrema(x, y, in_extrema):
    in_extrema['minX'] = min(in_extrema.get('minX'), x)
    in_extrema['maxX'] = max(in_extrema.get('maxX'), x)
    in_extrema['minY'] = min(in_extrema.get('minY'), y)
    in_extrema['maxY'] = max(in_extrema.get('maxY'), y)



nodes = []
edges = []
node_map = {}

def ReLU(x):
    return max(0, x)

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def node_edge_init(file):
    raw = gpd.read_file(file)
    ID = -1
    for (index, row) in raw.geometry.items():
        ID += 1
        if row.geom_type == "LineString":
            for i in range(len(row.coords) - 1):
                edge = Edge(list(row.coords[i]), list(row.coords[i + 1]))
                edges.append(edge)

            for x, y in row.coords:
                findExtrema(x, y, extrema)
                node_map[(int(x), int(y))] = Node(int(x), int(y), ID)

    nodes.extend(node_map.values())


    for node in nodes: # adjust to fit screen dimensions
        node.x -= extrema.get('minX')
        node.y -= extrema.get('minY')
        node.x = ReLU(int(node.x))
        node.y = ReLU(int(node.y))
        findExtrema(node.x, node.y, extrema2)

    for node in nodes:
        node.x *= w / (extrema2.get('maxX'))
        node.y = h - (node.y * h / extrema2.get('maxY'))
        node.update_p()
        findExtrema(node.x, node.y, extrema3)
        node.x = ReLU(int(node.x))
        node.y = ReLU(int(node.y))
        node_map[(node.x, node.y)] = node

    for edge in edges:
        for i in range(2):
            edge.points[i][0] -= extrema.get('minX')
            edge.points[i][1] -= extrema.get('minY')
            if edge.points[i][0] < 0:
                edge.points[i][0] = 0
            if edge.points[i][1] < 0:
                edge.points[i][1] = 0
            edge.points[i][0] *= w / (extrema2.get('maxX'))
            edge.points[i][1] = h - (edge.points[i][1] * h / extrema2.get('maxY'))
            edge.points[i][0] = ReLU(int(edge.points[i][0]))
            edge.points[i][1] = ReLU(int(edge.points[i][1]))
            edge.points[i] = tuple(edge.points[i])






    for edge in edges:
        p1 = tuple(edge.p1)
        p2 = tuple(edge.p2)
        print(p1, p2)

        if p1 in node_map and p2 in node_map:
            node_map[p1].neighbors.append(node_map[p2])
            node_map[p2].neighbors.append(node_map[p1])



    print(extrema3)


node_edge_init("data/South_Clear_Creek_Roads.shp")

#print([node.to_string() + '\n' for node in nodes])

def write_intersections():
    csv_file = open('data/intersections.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    for (i, edge1), (j, edge2) in combinations(enumerate(edges), 2):
        inters = line_intersection((edge1.p1, edge1.p2), (edge2.p1, edge2.p2))
        if node_map.get(inters):
            continue
        if inters:
            writer.writerow([inters[0], inters[1]])

    csv_file.close()
    print("done")


bot = Robot(1, nodes[150], nodes[1000], nodes)
print(bot.robot.x, bot.robot.y)




def paint():
    screen.fill((255, 255, 255))
    for edge in edges:
        p1 = (int(edge.p1[0]), int(edge.p1[1]))
        p2 = (int(edge.p2[0]), int(edge.p2[1]))
        pygame.draw.line(screen, (173, 216, 230), p1, p2, 2)

#    draw_line(screen, (200, 200, 200), (50, 20), (50, 80), 5)


    bot.update()
    print(bot.robot.x, bot.robot.y)

    pygame.draw.rect(screen, (50, 30, 30), bot.robot)
    #draw_tri(screen, bot.robot.x, bot.robot.y, 20)
    pygame.display.flip()

    clock.tick(60)

# def draw_tri(screen, x, y, size):
#     base_left = (x - size // 2, y)  # Bottom-left
#     base_right = (x + size // 2, y)  # Bottom-right
#     tip = (x, y + size)
#     pygame.draw.polygon(screen, (30, 140, 60), [base_left, base_right, tip])

write_intersections()

running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#         if event.type == pygame.VIDEORESIZE:
#             # There's some code to add back window content here.
#             surface = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
#     paint()








pygame.quit()
