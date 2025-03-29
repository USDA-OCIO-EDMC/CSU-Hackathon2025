# adjacent_strategy.py
"""
This file demonstrates an adjacent strategy for processing a DEM and initial grid
to build a game map, compute a placeholder mask, run A* search, simulate player movement,
and overlay the computed path onto the initial grid.

This file is for educational purposes only.
"""

import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'  # Silence audio warnings

from PIL import Image, ImageDraw
import numpy as np
import heapq
import random

# Disable decompression bomb warnings (if you trust the image)
Image.MAX_IMAGE_PIXELS = None

### Helper functions ###

def label_max_diff(data, threshold):
    """
    For each cell in a 2D array 'data', compute the maximum absolute difference
    between the cell and its cardinal neighbors (N, S, E, W). If the max difference 
    exceeds the threshold, label the cell as 1; otherwise, label it 0.
    
    Pads the array with edge values to handle border cells.
    """
    padded = np.pad(data, pad_width=1, mode='edge')
    center = padded[1:-1, 1:-1]
    north  = padded[:-2, 1:-1]
    south  = padded[2:, 1:-1]
    east   = padded[1:-1, 2:]
    west   = padded[1:-1, :-2]
    diff_north = np.abs(center - north)
    diff_south = np.abs(center - south)
    diff_east  = np.abs(center - east)
    diff_west  = np.abs(center - west)
    max_diff = np.maximum.reduce([diff_north, diff_south, diff_east, diff_west])
    labels = (max_diff > threshold).astype(np.int32)
    return labels

def astar_search(game_map, start, targets):
    """
    A* search on a numeric grid with the following codes:
      0 = background, 1 = path, 2 = wall, 3 = reward.
    Walls (2) are impassable.
    Returns the optimal path as a list of (row, col) tuples.
    """
    rows, cols = game_map.shape

    def heuristic(cell):
        r, c = cell
        return min(abs(r - tr) + abs(c - tc) for tr, tc in targets)

    open_set = []
    start_h = heuristic(start)
    heapq.heappush(open_set, (start_h, 0, start))
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_set:
        f, g, current = heapq.heappop(open_set)
        if current in targets:
            # Reconstruct path from start to current
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        visited.add(current)
        r, c = current
        # Explore neighbors (N, S, E, W)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if game_map[nr, nc] == 2:  # wall is impassable
                    continue
                neighbor = (nr, nc)
                tentative_g = g + 1
                if neighbor in visited and tentative_g >= g_score.get(neighbor, float('inf')):
                    continue
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    return None

def simulate_player(game_map, start=None, reward=None):
    """
    Simulate a player moving on a numeric game map.
    Codes:
      0 = background, 1 = path, 2 = wall, 3 = reward.
    If start or reward positions are not provided, they are chosen at random
    from non-wall cells.
    Returns the path (a list of (row, col) coordinates).
    """
    rows, cols = game_map.shape

    # Helper to choose a random cell that's not a wall.
    def random_non_wall():
        valid = [(r, c) for r in range(rows) for c in range(cols) if game_map[r, c] != 2]
        return random.choice(valid) if valid else None

    if start is None:
        start = random_non_wall()
        print("Randomly chosen start:", start)
    if reward is None:
        reward = random_non_wall()
        print("Randomly chosen reward:", reward)
        game_map[reward] = 3  # mark reward cell
    else:
        game_map[reward] = 3  # ensure reward cell is marked

    if game_map[start] == 3:
        print(f"Spawned on reward at {start}. Final position: {start}")
        return [start]

    targets = [(r, c) for r in range(rows) for c in range(cols) if game_map[r, c] == 3]
    if not targets:
        print("No reward cell available!")
        return [start]

    path = astar_search(game_map, start, targets)
    if not path:
        print("No path found to the reward!")
        return [start]

    print("Calculated optimal path to reward:", path)
    return path

### Step 2: Create the Base Game Map from initial_grid.png ###
# Load the initial grid image (which has exactly two unique values)
img = Image.open('/notebooks/initial_grid.png').convert("L")
img_array = np.array(img)
unique_values = np.unique(img_array)
print("Unique pixel values in initial grid:", unique_values)

# Map the two unique values to 'background' and 'path'
mapping = {}
if len(unique_values) == 2:
    # In this example, assign the higher value to 'path'
    mapping[unique_values[1]] = 'path'
    mapping[unique_values[0]] = 'background'
else:
    raise ValueError("Expected exactly 2 unique values in the image")

# Convert to a numeric base game map: background -> 0, path -> 1.
base_game_map = np.array([[0 if mapping[pixel] == 'background' else 1 for pixel in row]
                          for row in img_array])
print("Base game map shape:", base_game_map.shape)

### Step 3: Load DEM and Compute the Placeholder Mask ###
# Load your DEM file (assumed to be the same shape as the initial grid)
dem_img = Image.open('/notebooks/South_Clear_Creek_BareEarth_DEM_1m.tif')
dem_data = np.array(dem_img)
# For placeholder detection, determine the maximum value in the DEM.
max_val = np.nanmax(dem_data)
print("DEM maximum value (assumed placeholder):", max_val)
# Create a binary mask where cells equal to the maximum are marked as invalid (1), else 0.
placeholder_mask = np.where(dem_data == max_val, 1, 0)
print("Placeholder mask shape:", placeholder_mask.shape)

### Step 4: Combine the Base Game Map and Placeholder Mask ###
# Overlay the placeholder mask onto the base game map:
# For each background cell (0), if placeholder_mask == 1 then mark it as a wall (2).
combined_map = base_game_map.copy()
rows, cols = combined_map.shape
for r in range(rows):
    for c in range(cols):
        if combined_map[r, c] == 0 and placeholder_mask[r, c] == 1:
            combined_map[r, c] = 2

print("Combined game map:")
print(combined_map)

### Step 5: Run the Simulation ###
# Optionally, define start and reward positions here.
# If set to None, they will be chosen randomly from non-wall cells.
start_loc = None    # e.g., (0, 0) or None for random
reward_loc = None   # e.g., (combined_map.shape[0]-1, combined_map.shape[1]-1) or None for random

path = simulate_player(combined_map, start=start_loc, reward=reward_loc)

### Step 6: Draw the Path on Top of the Initial Grid ###
# Reload the initial grid image and convert to RGB for drawing
grid_img = Image.open('/notebooks/initial_grid.png').convert("RGB")
draw = ImageDraw.Draw(grid_img)

# Convert path grid coordinates to pixel coordinates.
# Assuming each grid cell corresponds to one pixel; adjust scale if needed.
if path is not None and len(path) > 1:
    # Convert (row, col) to (x, y) i.e., (col, row)
    line_coords = [(c, r) for r, c in path]
    # Draw a light blue line (RGB (135, 206, 250)) with a width of 4 pixels
    draw.line(line_coords, fill=(135, 206, 250), width=4)

# Save the overlaid image
grid_img.save('/notebooks/path_overlay.png')
print("Path drawn and saved as '/notebooks/path_overlay.png'.")
