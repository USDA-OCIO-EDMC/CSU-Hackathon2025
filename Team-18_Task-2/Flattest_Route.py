import numpy as np
import rasterio
from heapq import heappop, heappush
from PIL import Image
import random
import os

def load_tif(file_path):
    with rasterio.open(file_path) as dataset:
        return dataset.read(1)

def parse_routes(route_data):
    route_pixels = [(y, x) for y in range(route_data.shape[0]) for x in range(route_data.shape[1]) if route_data[y, x] > 0.5]  # Adjust threshold if needed
    print("Parsed route pixels:", len(route_pixels))
    return np.array(route_pixels)  # Store as NumPy array for efficiency

def save_route_pixels(route_pixels, file_path="route_pixels.npy"):
    np.save(file_path, route_pixels)
    print(f"Route pixels saved to {file_path}")

def load_route_pixels(file_path="route_pixels.npy"):
    if os.path.exists(file_path):
        route_pixels = np.load(file_path)
        print(f"Loaded {len(route_pixels)} route pixels from {file_path}")
        return route_pixels
    return None  # File doesn't exist

def assign_weights(route_pixels, slope_data, grade_threshold):
    weights = {tuple(pixel): min(slope_data[pixel[0], pixel[1]], grade_threshold) for pixel in route_pixels}
    return weights

def dijkstra(route_pixels, weights, start, end):
    route_pixel_set = {tuple(p) for p in route_pixels}  # Convert to set for fast lookup

    if tuple(start) not in route_pixel_set or tuple(end) not in route_pixel_set:
        print("Error: Start or End point is not on a valid route!")
        return []
    
    pq = [(0, tuple(start))]
    costs = {tuple(start): 0}
    parents = {tuple(start): None}

    while pq:
        cost, current = heappop(pq)
        if current == tuple(end):
            path = []
            while current:
                path.append(current)
                current = parents[current]
            return path[::-1]
        
        y, x = current
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:  # Movement in 8 directions
            neighbor = (y + dy, x + dx)
            if neighbor in route_pixel_set and neighbor in weights:
                new_cost = cost + weights[neighbor]
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    heappush(pq, (new_cost, neighbor))
                    parents[neighbor] = current
    
    print("No valid path found between start and end points.")
    return []

def generate_output(route_data, best_path, flattest_path, start, end, output_file="output.png"):
    img = np.stack([route_data * 255] * 3, axis=-1)  # Convert grayscale to RGB
    
    for y, x in best_path:
        img[y, x] = [255, 0, 0]  # Red for best path
    for y, x in flattest_path:
        img[y, x] = [0, 255, 0]  # Green for flattest path
    
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if 0 <= start[0] + dy < img.shape[0] and 0 <= start[1] + dx < img.shape[1]:
                img[start[0] + dy, start[1] + dx] = [0, 0, 255]  # Blue for start
            if 0 <= end[0] + dy < img.shape[0] and 0 <= end[1] + dx < img.shape[1]:
                img[end[0] + dy, end[1] + dx] = [255, 255, 0]  # Yellow for end
    
    Image.fromarray(img.astype(np.uint8)).save(output_file)

def main():
    slope_data = load_tif("Upper_Willow_Creek_BareEarth_Hillshade_1m_1.tif")
    
    # Try loading cached route pixels
    route_pixels = load_route_pixels()
    
    if route_pixels is None:
        route_data = load_tif("Upper_Willow_Creek_Roads_Mask_1.tif")
        route_pixels = parse_routes(route_data)
        save_route_pixels(route_pixels)

    route_pixels = np.array([p for p in route_pixels if p[0] < slope_data.shape[0] and p[1] < slope_data.shape[1]])

    grade_threshold = 10.0  
    weights = assign_weights(route_pixels, slope_data, grade_threshold)
    
    start = tuple(random.choice(route_pixels))  
    end = tuple(random.choice(route_pixels))  
    
    best_path = dijkstra(route_pixels, weights, start, end)
    flattest_path = dijkstra(route_pixels, {tuple(p): 1 for p in route_pixels}, start, end)  
    
    route_data = load_tif("Upper_Willow_Creek_Roads_Mask_1.tif")  # Reload for visualization
    generate_output(route_data, best_path, flattest_path, start, end)

if __name__ == "__main__":
    main()