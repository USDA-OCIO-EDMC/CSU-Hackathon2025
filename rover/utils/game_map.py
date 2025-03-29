from PIL import Image
import numpy as np

def build_base_game_map(grid_path):
    """
    Loads the initial grid image, converts it to grayscale, and maps the two unique
    pixel values to a numeric game map: background (0) and path (1).
    """
    
    img = Image.open(grid_path).convert("L")
    img_array = np.array(img)
    unique_values = np.unique(img_array)
    print("Unique pixel values in initial grid:", unique_values)

    mapping = {}
    if len(unique_values) == 2:
        # For example, assign the higher value to 'path' and the lower to 'background'
        mapping[unique_values[1]] = 'path'
        mapping[unique_values[0]] = 'background'
    else:
        raise ValueError("Expected exactly 2 unique values in the image")
    
    base_game_map = np.array([[0 if mapping[pixel] == 'background' else 1 
                               for pixel in row] for row in img_array])
    print("Base game map shape:", base_game_map.shape)
    return base_game_map

def combine_map(base_game_map, placeholder_mask):
    """
    Overlays the placeholder mask onto the base game map.
    For every background cell (0) that has a placeholder (mask==1), mark it as a wall (2).
    """
    combined_map = base_game_map.copy()
    rows, cols = combined_map.shape
    for r in range(rows):
        for c in range(cols):
            if combined_map[r, c] == 0 and placeholder_mask[r, c] == 1:
                combined_map[r, c] = 2  # Mark as wall
    print("Combined game map created.")
    return combined_map
