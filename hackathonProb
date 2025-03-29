# Final Hackathon Created on 29 mar
# author - vishnu
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import heapq
import math
import random
import matplotlib
matplotlib.use('TkAgg')

# === Euclidean Distance ===
def euclidean(p1, p2):
    return math.dist(p1, p2)

# === Enhanced Bidirectional A* with F/G/H simulation and discontinuity fallback ===
def bidirectional_a_star_discontinuity_aware(graph, start, goal, ax):
    fwd_open = [(0, start)]
    bwd_open = [(0, goal)]

    fwd_g = {n: float('inf') for n in graph.nodes}
    bwd_g = {n: float('inf') for n in graph.nodes}
    fwd_g[start] = 0
    bwd_g[goal] = 0

    fwd_parent = {}
    bwd_parent = {}

    fwd_visited = set()
    bwd_visited = set()

    meeting_node = None
    best_path_len = float('inf')

    open_f_marker, = ax.plot([], [], 'ro', label='Forward Frontier', markersize=3)
    open_b_marker, = ax.plot([], [], 'bo', label='Backward Frontier', markersize=3)
    f_open_x, f_open_y, b_open_x, b_open_y = [], [], [], []

    f_text = ax.text(0.01, 0.2, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

    while fwd_open and bwd_open:
        _, fwd_node = heapq.heappop(fwd_open)
        _, bwd_node = heapq.heappop(bwd_open)

        fwd_visited.add(fwd_node)
        bwd_visited.add(bwd_node)

# appending the nodes current nodes (start and goal) to the open list because initially they are not explored yet 
# becuase they are tuple, we can store them using their index [0] and [1] ==> their x and y coordinate.

        f_open_x.append(fwd_node[0])
        f_open_y.append(fwd_node[1])
        b_open_x.append(bwd_node[0])
        b_open_y.append(bwd_node[1])
        open_f_marker.set_data(f_open_x, f_open_y)
        open_b_marker.set_data(b_open_x, b_open_y)

        f = fwd_g[fwd_node] + euclidean(fwd_node, goal)
        b = bwd_g[bwd_node] + euclidean(bwd_node, start)
        f_text.set_text(f"FWD: G={fwd_g[fwd_node]:.1f}, H={euclidean(fwd_node, goal):.1f}, F={f:.1f}\n"
                         f"BWD: G={bwd_g[bwd_node]:.1f}, H={euclidean(bwd_node, start):.1f}, F={b:.1f}")

        plt.draw()
        plt.pause(0.0005)

        # Forward Expansion
        fwd_neighbors = list(graph.neighbors(fwd_node))
        unexplored_fwd = [n for n in fwd_neighbors if n not in fwd_parent]
        for neighbor in unexplored_fwd if unexplored_fwd else fwd_neighbors:
            temp_g = fwd_g[fwd_node] + graph[fwd_node][neighbor]['weight']
            if temp_g < fwd_g[neighbor]:
                fwd_g[neighbor] = temp_g
                heapq.heappush(fwd_open, (temp_g + euclidean(neighbor, goal), neighbor))
                fwd_parent[neighbor] = fwd_node
                if neighbor in bwd_g and bwd_g[neighbor] < float('inf'):
                    total = temp_g + bwd_g[neighbor]
                    if total < best_path_len:
                        best_path_len = total
                        meeting_node = neighbor

        # Backward Expansion
        bwd_neighbors = list(graph.neighbors(bwd_node))
        unexplored_bwd = [n for n in bwd_neighbors if n not in bwd_parent]
        for neighbor in unexplored_bwd if unexplored_bwd else bwd_neighbors:
            temp_g = bwd_g[bwd_node] + graph[bwd_node][neighbor]['weight']
            if temp_g < bwd_g[neighbor]:
                bwd_g[neighbor] = temp_g
                heapq.heappush(bwd_open, (temp_g + euclidean(neighbor, start), neighbor))
                bwd_parent[neighbor] = bwd_node
                if neighbor in fwd_g and fwd_g[neighbor] < float('inf'):
                    total = temp_g + fwd_g[neighbor]
                    if total < best_path_len:
                        best_path_len = total
                        meeting_node = neighbor

        if meeting_node:
            path = []
            node = meeting_node
            while node in fwd_parent:
                path.append(node)
                node = fwd_parent[node]
            path.append(start)
            path.reverse()
            node = meeting_node
            while node in bwd_parent:
                node = bwd_parent[node]
                path.append(node)
            return path

    return None

# === Load Shapefile using geo pandas ===
shapefile_path = "/Users/vishnujordan/Downloads/Roads_Boundary/South_Clear_Creek_Roads.shp"
gdf = gpd.read_file(shapefile_path)

# === Build Graph using networkX library ===
G = nx.Graph()
for _, row in gdf.iterrows():
    geom = row.geometry
    if isinstance(geom, LineString):
        coords = list(geom.coords)
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            dist = Point(p1).distance(Point(p2))
            G.add_edge(p1, p2, weight=dist)
def segments_intersect(a1, a2, b1, b2):
    def ccw(p1, p2, p3):
        return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])

    if (ccw(a1, b1, b2) != ccw(a2, b1, b2)) and (ccw(a1, a2, b1) != ccw(a1, a2, b2)):
        x1, y1 = a1
        x2, y2 = a2
        x3, y3 = b1
        x4, y4 = b2
        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if denom == 0:
            return None
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return (round(px, 6), round(py, 6))
    return None


# Generally, the intersecting edges are not sharing the nodes.. so the code won't be able to traverse
# so its recommended (sometimes necessary) to build the intersection points and create additional nodes on them and add them to the graph map (with additional nodes with corresponding edges)
def add_intersection_nodes(graph):
    edges = list(graph.edges)
    new_nodes = set()
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            p1, p2 = edges[i]
            q1, q2 = edges[j]
            inter = segments_intersect(p1, p2, q1, q2)
            if inter and inter not in graph.nodes:
                new_nodes.add(inter)
    for node in new_nodes:
        graph.add_node(node)
    print(f" Added {len(new_nodes)} intersection nodes")
    return list(new_nodes)


####
# Call this after building the graph
intersection_nodes = add_intersection_nodes(G)
        

# === Choose Start & Goal ===



# # === Choose Start & Goal with length between 250–500
# nodes = list(G.nodes)
# start, goal = None, None
# for _ in range(1000):
#     s, g = random.sample(nodes, 2)
#     if nx.has_path(G, s, g):
#         p = nx.shortest_path(G, s, g, weight='weight')
#         if 500 <= len(p) <= 600:
#             start, goal = s, g
#             break

# if not start:
#     raise Exception(" Couldn't find suitable start and goal.")

# print(f"Start Node: {start}")
# print(f"Goal Node: {goal}")





# === Set Start and Goal ===
# fixed_start = (434189.6902, 4388844.3178)
# fixed_goal = (436216.7882, 4390206.0985)

fixed_start = (433722.1777999997, 4385888.024700001)
fixed_goal = (437565.85309999995, 4392515.620100001)

start = min(G.nodes, key=lambda n: Point(n).distance(Point(fixed_start)))
goal = min(G.nodes, key=lambda n: Point(n).distance(Point(fixed_goal)))
  
print(f"Start Node: {start}")
print(f"Goal Node: {goal}")

####



######

# # === Fixed Start and Goal Coordinates ===  
# # this is for demonstration
# # needs to toggled off / commented out in the actual execution
# fixed_start = (439707.50629999954, 4392610.8367)
# fixed_goal = (437316.25009999983, 4392078.9958999995)
# # === Function to snap to nearest node in the graph ===
# # def get_nearest_node(graph, point):
# #     return min(graph.nodes, key=lambda n: Point(n).distance(Point(point)))

# # === Snap coordinates to graph nodes ===
# start = get_nearest_node(G, fixed_start)
# goal = get_nearest_node(G, fixed_goal)

# === Check connectivity ===



######


######

# if not nx.has_path(G, start, goal):
#     raise Exception(" No path exists between start and goal.")

# # === Log snapped node coordinates ===
# print(f" Start Node: {start}")
# print(f" Goal Node: {goal}")

# ######





# === Visualize ===
plt.ion()
fig, ax = plt.subplots(figsize=(12, 12))
for _, row in gdf.iterrows():
    geom = row.geometry
    if isinstance(geom, LineString):
        x, y = geom.xy
        ax.plot(x, y, color='lightgray', linewidth=0.5)

ax.plot(start[0], start[1], 'go', label='Start', markersize=8)
ax.plot(goal[0], goal[1], 'bo', label='Goal', markersize=8)
ax.set_title("Bidirectional A* with Discontinuity Awareness & with F value (G value + H value Simulation)")
ax.legend()
ax.axis('equal')

# === Run and Animate ===
path = bidirectional_a_star_discontinuity_aware(G, start, goal, ax)
if path:
    x_vals = [pt[0] for pt in path]
    y_vals = [pt[1] for pt in path]
    ax.plot(x_vals, y_vals, 'k-', linewidth=2, label='Final Path')
    print(f"Path found with {len(path)} steps")
else:
    print(" No path found")

plt.ioff()
plt.show()
