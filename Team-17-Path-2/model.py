import os
import time
import cv2
import numpy as np
import pygame
import heapq
import math
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
 
# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
 
DISPLAY_MAX_DIM = 800  # Max width/height for display
 
def load_tif_mask_color(file_path):
    """
    Loads a TIFF file and creates a binary mask where roads are represented as 1s (white).
    Returns:
      color_img (np.uint8): The original color image (3-channel BGR).
      mask_bw   (np.uint8): A 2D binary mask (0 or 255) for A*, where
                            roads are white=255 (traversable).
    """
    # Try multiple approaches to load the image
    color_img = None
   
    # Standard OpenCV approach
    color_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
   
    if color_img is None:
        # Try alternative approach - sometimes needed for specialized TIFF formats
        try:
            color_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            # If image has more than 3 channels (like RGBA), keep only RGB
            if color_img is not None and color_img.ndim > 2 and color_img.shape[2] > 3:
                color_img = color_img[:, :, :3]
        except:
            pass
   
    # If still not loaded, try with GDAL if available
    if color_img is None:
        try:
            from osgeo import gdal
            dataset = gdal.Open(file_path)
            if dataset:
                bands = []
                for i in range(1, min(4, dataset.RasterCount + 1)):  # Get up to 3 bands (RGB)
                    band = dataset.GetRasterBand(i)
                    bands.append(band.ReadAsArray())
               
                if bands:  # If we have at least one band
                    if len(bands) == 3:
                        color_img = cv2.merge(bands)
                    elif len(bands) == 1:
                        color_img = cv2.cvtColor(bands[0].astype(np.uint8), cv2.COLOR_GRAY2BGR)
        except ImportError:
            print("[INFO] GDAL not available for advanced TIFF handling")
        except Exception as e:
            print(f"[WARN] GDAL attempt failed: {str(e)}")
   
    if color_img is None:
        raise IOError(f"Could not load image from {file_path}")
   
    # Convert to grayscale
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
   
    # Try adaptive thresholding first
    try:
        # Enhance contrast if needed
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
       
        # Apply adaptive threshold (better for images with varying illumination)
        mask_bw = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
    except:
        # Fall back to Otsu's thresholding method (automatically determines best threshold)
        _, mask_bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
   
    # Save the mask for debugging
    cv2.imwrite('debug_mask.png', mask_bw)
   
    # Log info about the loaded image
    print(f"[DEBUG] Loaded image shape: {color_img.shape}")
    print(f"[DEBUG] Number of white pixels in mask: {np.sum(mask_bw > 127)}")
    print(f"[DEBUG] White pixel percentage: {(np.sum(mask_bw > 127) / (mask_bw.shape[0] * mask_bw.shape[1]) * 100):.2f}%")
   
    return color_img, mask_bw
 
def make_display_image(color_img):
    """
    Downscale the color_img if necessary so that
    neither width nor height exceeds DISPLAY_MAX_DIM.
    Returns:
      display_img, scale_x, scale_y
    """
    h, w = color_img.shape[:2]
    if h <= DISPLAY_MAX_DIM and w <= DISPLAY_MAX_DIM:
        return color_img, 1.0, 1.0
    else:
        scale = min(DISPLAY_MAX_DIM / float(h), DISPLAY_MAX_DIM / float(w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        display_img = cv2.resize(color_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scale_x = new_w / float(w)
        scale_y = new_h / float(h)
        return display_img, scale_x, scale_y
 
def snap_to_white(mask_bw, x, y, search_radius=50):
    """
    Ensures (x, y) is on a white (traversable) pixel by finding the
    nearest white pixel in mask_bw. Returns (xw, yw).
    Uses a more efficient radius-based search.
    """
    rows, cols = mask_bw.shape[:2]
    x, y = int(x), int(y)
   
    # If (x, y) is in-bounds and already white, return it
    if 0 <= y < rows and 0 <= x < cols and mask_bw[y, x] > 127:
        return (x, y)
   
    # Search in a window first for efficiency
    min_y = max(0, y - search_radius)
    max_y = min(rows, y + search_radius + 1)
    min_x = max(0, x - search_radius)
    max_x = min(cols, x + search_radius + 1)
   
    window = mask_bw[min_y:max_y, min_x:max_x]
    white_y, white_x = np.where(window > 127)
   
    if len(white_y) > 0:
        # Find closest white pixel within window
        distances = (white_y - (y - min_y))**2 + (white_x - (x - min_x))**2
        closest_idx = np.argmin(distances)
       
        # Convert back to original image coordinates
        closest_y = white_y[closest_idx] + min_y
        closest_x = white_x[closest_idx] + min_x
        return (closest_x, closest_y)
   
    # If no white pixels in search radius, expand to full image search
    print(f"[WARN] No white pixels in radius {search_radius}. Expanding to full image search.")
    white_y, white_x = np.where(mask_bw > 127)
   
    if len(white_y) == 0:
        raise ValueError("No white pixels found in the entire mask!")
   
    # Find closest white pixel in entire image
    distances = (white_y - y)**2 + (white_x - x)**2
    closest_idx = np.argmin(distances)
   
    return (white_x[closest_idx], white_y[closest_idx])
 
def generate_trail_mask(height, width, num_trails=5, steps=600,
                        thickness=3, min_step_length=3, max_step_length=7,
                        angle_change_range=np.pi/8):
    mask = np.zeros((height, width), dtype=np.uint8)
    for _ in range(num_trails):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        angle = np.random.uniform(0, 2 * np.pi)
        path = [(x, y)]
        for _ in range(steps):
            angle += np.random.uniform(-angle_change_range, angle_change_range)
            step_length = np.random.randint(min_step_length, max_step_length)
            dx = int(np.cos(angle) * step_length)
            dy = int(np.sin(angle) * step_length)
            x = np.clip(x + dx, 0, width - 1)
            y = np.clip(y + dy, 0, height - 1)
            path.append((x, y))
        for i in range(1, len(path)):
            cv2.line(mask, path[i-1], path[i], color=255, thickness=thickness)
    return mask
 
def astar(grid, start, goal):
    """
    A* on a grid of True/False for traversable/blocked (start, goal as (row, col)).
    Returns path as list of (row, col) coordinates.
    """
    # Define 8-connected neighborhood
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
   
    def heuristic(a, b):
        # Euclidean distance heuristic
        return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
   
    # Priority queue for open set, f-score as priority
    open_set = []
    heapq.heappush(open_set, (0, start))
   
    # Dict to track where each node came from
    came_from = {}
   
    # g_score: cost from start to current node
    g_score = {start: 0}
    steps_counter = 0
   
    while open_set:
        steps_counter += 1
        current_f, current = heapq.heappop(open_set)
 
        if steps_counter % 1000 == 0:
            print(f"[A* Debug] Step {steps_counter}, current={current}, open_set size={len(open_set)}")
 
        # Check if we've reached the goal
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            print(f"[INFO] A* found path in {steps_counter} expansions with length {len(path)}")
            return path
       
        # Explore all neighbors
        for dx, dy in neighbors:
            nr, nc = current[0] + dx, current[1] + dy
           
            # Check if the neighbor is in bounds and traversable
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if not grid[nr, nc]:  # If pixel is not traversable (black)
                    continue
               
                # Cost is g_score + distance to neighbor
                # For diagonal neighbors, distance is sqrt(2)
                # For cardinal neighbors, distance is 1
                if abs(dx) + abs(dy) == 2:  # Diagonal
                    move_cost = 1.414  # sqrt(2)
                else:  # Cardinal
                    move_cost = 1.0
               
                tentative_g = g_score[current] + move_cost
               
                # If we have a better path to this neighbor
                if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    f = tentative_g + heuristic((nr, nc), goal)
                    heapq.heappush(open_set, (f, (nr, nc)))
   
    # If we're here, no path was found
    print("[WARN] A* failed to find a path!")
    return None
 
def animate_astar_path(optimal_path_astar, display_img, start_pt, end_pt, scale_x, scale_y):
    pygame.init()
    disp_h, disp_w = display_img.shape[:2]
    screen = pygame.display.set_mode((disp_w, disp_h))
    pygame.display.set_caption("A* Path Visualization (Scaled)")
    clock = pygame.time.Clock()
    running = True
    path_index = 0
   
    disp_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
 
    scaled_path = []
    for (x, y) in optimal_path_astar:
        sx = int(x * scale_x)
        sy = int(y * scale_y)
        scaled_path.append((sx, sy))
 
    scaled_start = (int(start_pt[0] * scale_x), int(start_pt[1] * scale_y))
    scaled_end   = (int(end_pt[0]   * scale_x), int(end_pt[1]   * scale_y))
 
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
       
        surface_img = pygame.surfarray.make_surface(disp_img_rgb)
        screen.blit(surface_img, (0, 0))
       
        pygame.draw.lines(screen, (255, 255, 255), False, scaled_path, 2)
        pygame.draw.circle(screen, (0, 255, 0), scaled_start, 7)
        pygame.draw.circle(screen, (0, 0, 255), scaled_end, 7)
       
        if path_index < len(scaled_path):
            pygame.draw.circle(screen, (255, 0, 0), scaled_path[path_index], 5)
            path_index += 1
        else:
            pygame.time.delay(1500)
            running = False
       
        pygame.display.flip()
        clock.tick(60)
   
    pygame.quit()
 
def animate_ml_rollout(path, display_img, start_pt, end_pt, optimal_path_astar, scale_x, scale_y):
    pygame.init()
    disp_h, disp_w = display_img.shape[:2]
    screen = pygame.display.set_mode((disp_w, disp_h))
    pygame.display.set_caption("ML Best Rollout (Scaled)")
    clock = pygame.time.Clock()
    path_index = 0
    running = True
   
    disp_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
 
    scaled_astar = []
    for (x, y) in optimal_path_astar:
        sx = int(x * scale_x)
        sy = int(y * scale_y)
        scaled_astar.append((sx, sy))
 
    scaled_start = (int(start_pt[0] * scale_x), int(start_pt[1] * scale_y))
    scaled_end   = (int(end_pt[0]   * scale_x), int(end_pt[1]   * scale_y))
 
    scaled_rollout = []
    for (x, y) in path:
        sx = int(x * scale_x)
        sy = int(y * scale_y)
        scaled_rollout.append((sx, sy))
 
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
 
        surface_img = pygame.surfarray.make_surface(disp_img_rgb)
        screen.blit(surface_img, (0, 0))
       
        pygame.draw.lines(screen, (255, 255, 255), False, scaled_astar, 2)
        pygame.draw.circle(screen, (0, 255, 0), scaled_start, 7)
        pygame.draw.circle(screen, (0, 0, 255), scaled_end, 7)
       
        if path_index < len(scaled_rollout):
            pygame.draw.circle(screen, (255, 0, 0), scaled_rollout[path_index], 5)
            path_index += 1
        else:
            pygame.time.delay(1500)
            running = False
       
        pygame.display.flip()
        clock.tick(60)
   
    pygame.quit()
 
def find_diverse_start_end_points(mask_bw, min_distance=100):
    """
    Find diverse start and end points on the white mask
    that are sufficiently far apart from each other
    """
    white_y, white_x = np.where(mask_bw > 127)
   
    if len(white_y) < 2:
        raise ValueError("[ERROR] Not enough white pixels (traversable).")
   
    # Find opposite sides of the mask by using distance from the center
    h, w = mask_bw.shape[:2]
    center_y, center_x = h // 2, w // 2
   
    # Calculate distance of each white point from center
    distances = (white_y - center_y)**2 + (white_x - center_x)**2
   
    # Sort points by their angle from center
    angles = np.arctan2(white_y - center_y, white_x - center_x)
    sorted_indices = np.argsort(angles)
   
    # Try to find points at opposite sides (approximately 180 degrees apart)
    best_start_idx = sorted_indices[0]
    best_end_idx = None
    best_distance = 0
   
    for i in range(len(sorted_indices)):
        idx = sorted_indices[i]
        distance = (white_x[idx] - white_x[best_start_idx])**2 + (white_y[idx] - white_y[best_start_idx])**2
        if distance > best_distance and distance >= min_distance**2:
            best_end_idx = idx
            best_distance = distance
   
    if best_end_idx is None:
        # Fall back to leftmost and rightmost points if we couldn't find well-separated points
        x_indices = np.argsort(white_x)
        best_start_idx = x_indices[0]
        best_end_idx = x_indices[-1]
 
    # Convert to (x,y) format    
    start_pt = (int(white_x[best_start_idx]), int(white_y[best_start_idx]))
    end_pt = (int(white_x[best_end_idx]), int(white_y[best_end_idx]))
   
    return start_pt, end_pt
 
if __name__ == "__main__":
    user_tif_path = input("Enter the path to a .tif file (leave blank to use random generation): ").strip()
 
    print("[INFO] Starting mask acquisition...")
    start_time = time.time()
 
    color_img = None
    mask_bw = None
 
    if user_tif_path:
        try:
            if not os.path.exists(user_tif_path):
                raise IOError(f"File does not exist: {user_tif_path}")
            color_img, mask_bw = load_tif_mask_color(user_tif_path)
            print(f"[INFO] Loaded color .tif from {user_tif_path}")
           
            # Verify mask has sufficient white pixels
            white_pixel_count = np.sum(mask_bw > 127)
            if white_pixel_count < 100:
                print(f"[WARN] Very few white pixels ({white_pixel_count}). Try inverting the mask.")
                mask_bw = cv2.bitwise_not(mask_bw)
                print(f"[INFO] After inversion: {np.sum(mask_bw > 127)} white pixels")
        except Exception as e:
            print(f"[WARN] Could not load .tif file ({e}). Falling back to random mask.")
            mask_bw = generate_trail_mask(400, 400)
            color_img = cv2.cvtColor(mask_bw, cv2.COLOR_GRAY2BGR)
    else:
        print("[INFO] No .tif file specified. Using random mask generation.")
        mask_bw = generate_trail_mask(400, 400)
        color_img = cv2.cvtColor(mask_bw, cv2.COLOR_GRAY2BGR)
   
    # Save binary mask for debugging
    cv2.imwrite('mask_binary.png', mask_bw)
   
    display_img, scale_x, scale_y = make_display_image(color_img)
 
    height, width = mask_bw.shape[:2]
    print(f"[INFO] Original color_img size: {color_img.shape[1]}x{color_img.shape[0]}")
    print(f"[INFO] mask_bw size:           {width}x{height}")
    print(f"[INFO] display_img size:       {display_img.shape[1]}x{display_img.shape[0]}")
    print(f"[INFO] scale_x={scale_x:.4f}, scale_y={scale_y:.4f}. Time = {time.time() - start_time:.2f} s\n")
 
    # ---------- A* Setup ----------
    print("[INFO] Finding start/goal and running A*...")
    start_time = time.time()
    grid = (mask_bw > 127)
   
    # Try to find diverse start and end points
    try:
        start_pt, end_pt = find_diverse_start_end_points(mask_bw)
        print(f"[INFO] Found diverse start/end points: {start_pt}, {end_pt}")
    except Exception as e:
        print(f"[WARN] Could not find diverse points: {e}. Using simple leftmost/rightmost.")
        # Find leftmost & rightmost white pixels as fallback
        white_pixels = np.column_stack(np.where(mask_bw > 127))
        if len(white_pixels) < 2:
            raise ValueError("[ERROR] Not enough white pixels (traversable).")
       
        # leftmost & rightmost white pixels
        start_idx = np.argmin(white_pixels[:, 1])
        end_idx   = np.argmax(white_pixels[:, 1])
        # Proposed start_pt, end_pt in (x,y)
        start_pt = (int(white_pixels[start_idx, 1]), int(white_pixels[start_idx, 0]))
        end_pt   = (int(white_pixels[end_idx,   1]), int(white_pixels[end_idx,   0]))
   
    # Ensure start and end points are on white pixels
    start_pt = snap_to_white(mask_bw, start_pt[0], start_pt[1])  # returns (x, y)
    end_pt   = snap_to_white(mask_bw, end_pt[0],   end_pt[1])    # returns (x, y)
 
    print(f"[INFO] Final start_pt={start_pt}")
    print(f"[INFO] Final end_pt={end_pt}")
 
    # For A*, we need (row, col)
    astar_start = (start_pt[1], start_pt[0])  # (row, col)
    astar_goal  = (end_pt[1],   end_pt[0])
 
    optimal_path_astar = astar(grid, astar_start, astar_goal)
    if optimal_path_astar is None:
        raise ValueError("[ERROR] No path found via A*.")
   
    # Convert A* path to (x,y)
    optimal_path_astar = [(rc[1], rc[0]) for rc in optimal_path_astar]
    a_star_time = time.time() - start_time
    print(f"[INFO] A* done. Path length = {len(optimal_path_astar)}. Time = {a_star_time:.2f} s\n")
 
    # ---------- Animate A* (scaled for display) ----------
    print("[INFO] Animating A* path over scaled color image...")
    animate_astar_path(optimal_path_astar, display_img, start_pt, end_pt, scale_x, scale_y)
    print("[INFO] A* path animation complete.\n")
 
    # ---------- Prepare Data for ML ----------
    print("[INFO] Preparing sequence data for ML...")
    start_time = time.time()
 
    # Convert path to normalized coordinates
    norm_path = []
    for x, y in optimal_path_astar:
        norm_path.append([x/width, y/height])
    norm_path = np.array(norm_path, dtype=np.float32)
 
    # Create training data pairs: sequence -> next position
    seq_length = 5  # Use 5 consecutive points to predict the next one
    X_train = []
    y_train = []
 
    for i in range(len(norm_path) - seq_length):
        X_train.append(norm_path[i:i+seq_length])
        y_train.append(norm_path[i+seq_length])
 
    X_train = np.array(X_train)
    y_train = np.array(y_train)
 
    print(f"[INFO] Created {len(X_train)} training examples")
    print(f"[INFO] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
   
    # ---------- Build & Train Simple ML Model ----------
    print("[INFO] Building and training simple ML model...")
   
    # Build a simple LSTM model
    model = tf.keras.Sequential([
        layers.LSTM(64, input_shape=(seq_length, 2)),
        layers.Dense(32, activation='relu'),
        layers.Dense(2)  # Output: next position (x, y)
    ])
   
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
   
    # Train
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        verbose=1
    )
   
    print("[INFO] ML model training complete.")
 
    # ---------- ML Path Generation using A* Guidance ----------
    print("[INFO] Generating ML path with A* guidance...")
   
    # Use the beginning of the A* path to initialize
    init_sequence = norm_path[:seq_length].copy()
   
    # Start at the real start point
    current_pos = np.array(start_pt, dtype=np.float32)
    ml_path = [tuple(current_pos.astype(int))]
   
    # Set up guidance parameters
    goal_threshold = 5.0  # Distance to goal to consider "arrived"
    max_steps = 500
   
    # Keep track of progress
    best_dist_to_goal = np.linalg.norm(current_pos - np.array(end_pt, dtype=np.float32))
    consecutive_no_improvement = 0
   
    # Path generation loop
    for step in range(max_steps):
        # Get current normalized position
        norm_pos = current_pos / [width, height]
       
        # Update the sequence with current position
        init_sequence = np.vstack([init_sequence[1:], norm_pos])
       
        # Predict next position
        model_input = init_sequence.reshape(1, seq_length, 2)
        pred_norm = model.predict(model_input, verbose=0)[0]
       
        # Convert to image coordinates
        pred_pos = pred_norm * [width, height]
       
        # Keep predictions on the road
        pred_x, pred_y = int(pred_pos[0]), int(pred_pos[1])
        on_road = 0 <= pred_y < height and 0 <= pred_x < width and mask_bw[pred_y, pred_x] > 127
       
        if not on_road:
            pred_x, pred_y = snap_to_white(mask_bw, pred_x, pred_y)
            pred_pos = np.array([pred_x, pred_y], dtype=np.float32)
       
        # Limit step size to avoid jumps
        step_vec = pred_pos - current_pos
        step_size = np.linalg.norm(step_vec)
       
        if step_size > 5.0:  # Limit step size to 5 pixels
            step_vec = (step_vec / step_size) * 5.0
            pred_pos = current_pos + step_vec
            pred_x, pred_y = int(pred_pos[0]), int(pred_pos[1])
            pred_x, pred_y = snap_to_white(mask_bw, pred_x, pred_y)
            pred_pos = np.array([pred_x, pred_y], dtype=np.float32)
       
        # Update position
        current_pos = pred_pos
        ml_path.append(tuple(current_pos.astype(int)))
       
        # Check distance to goal
        dist_to_goal = np.linalg.norm(current_pos - np.array(end_pt, dtype=np.float32))
       
        if dist_to_goal < best_dist_to_goal:
            best_dist_to_goal = dist_to_goal
            consecutive_no_improvement = 0
            print(f"  [Step {step}] Improved distance to goal: {best_dist_to_goal:.2f}")
        else:
            consecutive_no_improvement += 1
       
        # If no improvement for a while, use A* guidance
        if consecutive_no_improvement >= 5:
            print(f"  [Step {step}] No improvement for 5 steps, applying A* guidance")
           
            # Find closest point on A* path
            min_dist = float('inf')
            closest_idx = 0
           
            for i, (x, y) in enumerate(optimal_path_astar):
                dist = np.linalg.norm(current_pos - np.array([x, y], dtype=np.float32))
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
           
            # Move along A* path
            target_idx = min(closest_idx + 5, len(optimal_path_astar) - 1)
            target_x, target_y = optimal_path_astar[target_idx]
           
            # Move to target position
            current_pos = np.array([target_x, target_y], dtype=np.float32)
            ml_path.append(tuple(current_pos.astype(int)))
           
            # Update sequence window with A* path points
            for i in range(seq_length):
                idx = min(target_idx + i, len(optimal_path_astar) - 1)
                x, y = optimal_path_astar[idx]
                init_sequence[i] = [x / width, y / height]
           
            consecutive_no_improvement = 0
       
        # Check if reached goal
        if dist_to_goal < goal_threshold:
            ml_path.append(end_pt)  # Add exact endpoint
            print(f"  [Step {step+1}] Reached goal! Final distance: {dist_to_goal:.2f}")
            break
       
        # If we're near the end of path generation, follow A* directly to goal
        if step >= max_steps - 20:
            print(f"  [Step {step}] Near max steps, following A* to goal")
           
            # Find closest point on A* path
            min_dist = float('inf')
            closest_idx = 0
           
            for i, (x, y) in enumerate(optimal_path_astar):
                dist = np.linalg.norm(current_pos - np.array([x, y], dtype=np.float32))
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
           
            # Add remaining A* path to ML path
            for i in range(closest_idx, len(optimal_path_astar)):
                ml_path.append(optimal_path_astar[i])
           
            # Add endpoint
            ml_path.append(end_pt)
            break
   
    # If we still haven't reached the goal, add the endpoint
    if ml_path[-1] != end_pt:
        ml_path.append(end_pt)
   
    print(f"[INFO] ML path generation complete. Path length: {len(ml_path)}")
 
    # ---------- Animate ML Rollout ----------
    print("[INFO] Animating ML rollout...")
    animate_ml_rollout(ml_path, display_img, start_pt, end_pt, optimal_path_astar, scale_x, scale_y)
    print("[INFO] ML rollout animation complete.\n")
 
    print("[INFO] Done! Exiting.")
   
   
 