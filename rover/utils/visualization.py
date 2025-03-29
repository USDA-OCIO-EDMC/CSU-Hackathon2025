from PIL import Image, ImageDraw

def draw_path_on_grid(grid_path, path, output_path, line_width=10, line_color=(135, 206, 250),
                      marker_radius=15, marker_color=(255, 0, 0)):
    """
    Draws the given path on top of the initial grid image and places a marker
    at the end of the path.
    
    Args:
        grid_path (str): Path to the initial grid image.
        path (list of tuple): List of (row, col) coordinates.
        output_path (str): Path to save the resulting image.
        line_width (int): Width of the drawn path line.
        line_color (tuple): Color of the path line (RGB).
        marker_radius (int): Radius of the marker circle at the end of the path.
        marker_color (tuple): Color of the marker (RGB).
    """
    grid_img = Image.open(grid_path).convert("RGB")
    draw = ImageDraw.Draw(grid_img)
    
    # Convert grid coordinates (row, col) to pixel coordinates (x, y)
    if path is not None and len(path) > 1:
        line_coords = [(c, r) for r, c in path]
        draw.line(line_coords, fill=line_color, width=line_width)
        
        # Draw marker at the end of the path (last coordinate)
        end_point = line_coords[-1]
        left_up = (end_point[0] - marker_radius, end_point[1] - marker_radius)
        right_down = (end_point[0] + marker_radius, end_point[1] + marker_radius)
        draw.ellipse([left_up, right_down], fill=marker_color)
    
    grid_img.save(output_path)
    print(f"Path drawn and saved as '{output_path}'.")
