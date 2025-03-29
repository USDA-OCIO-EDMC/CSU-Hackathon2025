import geopandas as gpd
import pygame
import networkx as nx
import math
from heapq import heappush, heappop
from shapely.geometry import LineString, Point

class RoadNetwork:
    def __init__(self, shapefile_path, screen_width=800, screen_height=600):
        # Load the shapefile
        self.roads = gpd.read_file(shapefile_path)
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Scaling factors for converting between geo and screen coordinates
        self.min_x, self.min_y, self.max_x, self.max_y = self.get_bounds()
        self.scale_x = self.screen_width / (self.max_x - self.min_x)
        self.scale_y = self.screen_height / (self.max_y - self.min_y)

        # Extract road coordinates and create graph
        self.graph = self.create_road_graph()
            
    def is_point_on_road(self, screen_x, screen_y, tolerance=10):
        """Check if a given screen point is on the road within a tolerance"""
        # Convert screen coordinates to geographic coordinates
        geo_x, geo_y = self.screen_to_geo(screen_x, screen_y)
    
        # Convert the point to the same CRS as the roads
        point_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([geo_x], [geo_y]), crs=self.roads.crs)
    
        # Check distance to each road segment
        for road in self.roads.geometry:
            distance = road.distance(point_gdf.geometry[0])
            if distance < tolerance:
                print(f"Distance to road: {distance}")
                return True
        return False
    
    def get_bounds(self):
        """Get the bounds of the road network"""
        bounds = self.roads.total_bounds
        return bounds[0], bounds[1], bounds[2], bounds[3]  # min_x, min_y, max_x, max_y
    
    def geo_to_screen(self, x, y):
        """Convert geographic coordinates to screen coordinates"""
        screen_x = int((x - self.min_x) * self.scale_x)
        # Flip y axis because Pygame's origin is top-left
        screen_y = int(self.screen_height - (y - self.min_y) * self.scale_y)
        return screen_x, screen_y
    
    def screen_to_geo(self, screen_x, screen_y):
        """Convert screen coordinates to geographic coordinates"""
        x = (screen_x / self.scale_x) + self.min_x
        y = ((self.screen_height - screen_y) / self.scale_y) + self.min_y
        return x, y

    def add_intersections(self, G):
        """Add intersection nodes and edges to the graph without splitting existing edges"""
        def lines_intersect(line1, line2):
            """Check if two line segments intersect"""
            def ccw(A, B, C):
                return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
            
            A, B = line1
            C, D = line2
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
        
        edges = list(G.edges(data=True))
        for i in range(len(edges)):
            for j in range(i+1, len(edges)):
                edge1 = edges[i]
                edge2 = edges[j]
                
                line1 = [G.nodes[edge1[0]]['pos'], G.nodes[edge1[1]]['pos']]
                line2 = [G.nodes[edge2[0]]['pos'], G.nodes[edge2[1]]['pos']]
                
                if lines_intersect(line1, line2):
                    intersection = LineString(line1).intersection(LineString(line2))
                    if isinstance(intersection, Point):
                        intersection_coords = (intersection.x, intersection.y)
                        intersection_node = f"{intersection_coords}"
                        
                        if intersection_node not in G:
                            G.add_node(intersection_node, pos=intersection_coords)
                        
                        # Initialize distances
                        distance1 = distance2 = distance3 = distance4 = 0

                        # Add new edges connecting the intersection node to the endpoints of edge1
                        if not G.has_edge(edge1[0], intersection_node):
                            distance1 = LineString(line1).project(intersection)
                            G.add_edge(edge1[0], intersection_node, weight=distance1)
                        if not G.has_edge(intersection_node, edge1[1]):
                            distance2 = LineString(line1).length - distance1
                            G.add_edge(intersection_node, edge1[1], weight=distance2)
                        
                        # Add new edges connecting the intersection node to the endpoints of edge2
                        if not G.has_edge(edge2[0], intersection_node):
                            distance3 = LineString(line2).project(intersection)
                            G.add_edge(edge2[0], intersection_node, weight=distance3)
                        if not G.has_edge(intersection_node, edge2[1]):
                            distance4 = LineString(line2).length - distance3
                            G.add_edge(intersection_node, edge2[1], weight=distance4)

    def create_road_graph(self):
        """Create a graph representation of the road network"""
        G = nx.Graph()
        
        # For each road segment (LineString) in the shapefile
        for idx, road in self.roads.iterrows():
            geometry = road.geometry
            
            # Handle both LineString and MultiLineString geometries
            if geometry.geom_type == 'LineString':
                geometries = [geometry]
            else:  # MultiLineString
                geometries = list(geometry.geoms)
                
            for line in geometries:
                coords = list(line.coords)
                
                # Add nodes and edges to the graph
                for i in range(len(coords)-1):
                    # Add nodes (intersections and endpoints)
                    node1 = f"{coords[i]}"
                    node2 = f"{coords[i+1]}"
                    
                    if node1 not in G:
                        G.add_node(node1, pos=coords[i])
                    if node2 not in G:
                        G.add_node(node2, pos=coords[i+1])
                    
                    # Add edge (road segment) with distance as weight
                    distance = math.sqrt((coords[i+1][0] - coords[i][0])**2 + 
                                        (coords[i+1][1] - coords[i][1])**2)
                    
                    # Add road attributes if available (optional)
                    attrs = {'weight': distance}
                    if 'type' in road:
                        attrs['road_type'] = road['type']
                    if 'speed' in road:
                        attrs['speed_limit'] = road['speed']
                    
                    G.add_edge(node1, node2, **attrs)
        
        # Check for intersections and add intersection nodes/edges
        self.add_intersections(G)
        
        print("Nodes in the graph:", G.nodes)
        print("Edges in the graph:", G.edges)
        return G
    
    def find_nearest_node(self, pos):
        """Find the closest node to the given position"""
        min_dist = float('inf')
        nearest_node = None
        nearest_pos = None
        
        for node in self.graph.nodes():
            node_pos = self.graph.nodes[node]['pos']
            dist = math.sqrt((pos[0] - node_pos[0])**2 + (pos[1] - node_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
                nearest_pos = node_pos
        
        return nearest_node, nearest_pos
    
    def heuristic(self, a, b):
        """Calculate straight-line (Euclidean) distance heuristic for A*"""
        pos_a = self.graph.nodes[a]['pos']
        pos_b = self.graph.nodes[b]['pos']
        return math.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)

    def astar_path(self, start_node, end_node):
        """
        A* algorithm implementation for finding the shortest path
        """
        if start_node == end_node:
            return [start_node]
            
        # Dictionary to store path costs from start
        g_score = {start_node: 0}
        
        # Dictionary to store estimated total cost
        f_score = {start_node: self.heuristic(start_node, end_node)}
        
        # Priority queue for open set
        open_set = [(f_score[start_node], start_node)]
        
        # Dictionary to reconstruct path
        came_from = {}
        
        # Nodes that have been processed
        closed_set = set()
        
        while open_set:
            _, current = heappop(open_set)
            
            if current == end_node:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            closed_set.add(current)
            
            # Check all neighbors
            for neighbor in self.graph.neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                edge_weight = self.graph[current][neighbor]['weight']
                tentative_g_score = g_score[current] + edge_weight
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # This path is better, record it
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, end_node)
                    
                    # Add to open set if not already there
                    for i, (_, node) in enumerate(open_set):
                        if node == neighbor:
                            break
                    else:
                        heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        return []
    
    def find_path(self, start_pos, end_pos):
        """Find the shortest path between two positions using A*"""
        # Find the closest nodes to start and end positions
        start_node,_ = self.find_nearest_node(start_pos)
        end_node,_ = self.find_nearest_node(end_pos)
        
        # Use A* algorithm to find shortest path
        try:
            path = self.astar_path(start_node, end_node)
            
            if not path:
                print("No path found between the points")
                return []
                
            # Extract coordinates from the path
            path_coords = [self.graph.nodes[node]['pos'] for node in path]
            return path_coords
        except Exception as e:
            print(f"Error finding path: {e}")
            return []
    
    def draw(self, screen):
        """Draw the road network on the Pygame screen"""
        # Draw all road segments
        for idx, road in self.roads.iterrows():
            geometry = road.geometry
            
            # Handle both LineString and MultiLineString geometries
            if geometry.geom_type == 'LineString':
                geometries = [geometry]
            else:  # MultiLineString
                geometries = list(geometry.geoms)
                
            for line in geometries:
                prev_point = None
                
                for x, y in line.coords:
                    screen_x, screen_y = self.geo_to_screen(x, y)
                    
                    # Draw point (intersection/endpoint)
                    pygame.draw.circle(screen, (100, 100, 100), (screen_x, screen_y), 2)
                    
                    # Draw line segment
                    if prev_point:
                        pygame.draw.line(screen, (200, 200, 200), prev_point, (screen_x, screen_y), 2)
                    
                    prev_point = (screen_x, screen_y)
    
    def draw_path(self, screen, path, color=(255, 0, 0)):
        """Draw the calculated path on the Pygame screen"""
        if not path or len(path) < 2:
            return
        
        # Draw path segments
        for i in range(len(path)-1):
            # Convert to screen coordinates
            x1, y1 = self.geo_to_screen(path[i][0], path[i][1])
            x2, y2 = self.geo_to_screen(path[i+1][0], path[i+1][1])
            
            # Draw line segment
            pygame.draw.line(screen, color, (x1, y1), (x2, y2), 3)
            
            # Draw waypoints
            pygame.draw.circle(screen, color, (x1, y1), 4)
        
        # Draw last waypoint
        last_x, last_y = self.geo_to_screen(path[-1][0], path[-1][1])
        pygame.draw.circle(screen, color, (last_x, last_y), 4)
        
    def draw_path_stats(self, screen, path, font, color=(255, 255, 255)):
        """Draw statistics about the path"""
        if not path or len(path) < 2:
            return
            
        # Calculate total path length
        total_length = 0
        for i in range(len(path)-1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            segment_length = math.sqrt(dx*dx + dy*dy)
            total_length += segment_length
            
        # Display path information
        text = font.render(f"Path length: {total_length:.2f} units", True, color)
        screen.blit(text, (10, self.screen_height - 30))

    def draw_destination(self, screen, destination, color=(0, 255, 0)):
        """Draw the destination marker on the Pygame screen"""
        if destination is not None:
            screen_x, screen_y = self.geo_to_screen(*destination)
            pygame.draw.circle(screen, color, (screen_x, screen_y), 6)

    def draw_edges(self, screen, color=(0, 0, 255)):
        """Draw all edges in the graph on the Pygame screen"""
        for edge in self.graph.edges:
            node1, node2 = edge
            pos1 = self.graph.nodes[node1]['pos']
            pos2 = self.graph.nodes[node2]['pos']
            screen_x1, screen_y1 = self.geo_to_screen(*pos1)
            screen_x2, screen_y2 = self.geo_to_screen(*pos2)
            pygame.draw.line(screen, color, (screen_x1, screen_y1), (screen_x2, screen_y2), 2)