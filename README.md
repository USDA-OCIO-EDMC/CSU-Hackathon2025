# 🛰️ LiDAR - Prompt 1 & RoverChallenge - Prompt 2

## 📍 Challenge 1: Road Segmentation from LiDAR Hillshade

This module tackles the task of detecting roads from hillshade LiDAR data using a custom-trained U-Net-based segmentation model.

### 👩‍💻 Team
- Oshin Tiwari - [oshintiwari0928@gmail.com](mailto:oshintiwari0928@gmail.com)
- Vishnu Jawahar - [vishnu.lvj@gmail.com](mailto:vishnu.lvj@gmail.com)
- Luke Burrough - [luke.burrough@colostate.edu](mailto:luke.burrough@colostate.edu)
- Samuel Chamberlain - [Samuel.Chamberlain@rams.colostate.edu](mailto:Samuel.Chamberlain@rams.colostate.edu)

### 📌 Key Features
- ✅ U-Net model for binary segmentation of roads from LiDAR hillshade input
- ✅ Inference function that slides over large images with overlap and stitching
- ✅ Threshold tuning and sigmoid sharpening for better mask quality
- ✅ Final prediction exported as GeoTIFF for vectorization
- ✅ MCC (Matthews Correlation Coefficient) based evaluation pipeline

### 🛠️ Tools & Libraries
Python, PyTorch, Rasterio, NumPy, Matplotlib, Scikit-learn

### 📂 Files
- `Prompt 1 Solution.ipynb` – Full notebook for segmentation
- `Upper_Willow_Creek_BareEarth_Hillshade_1m_1_uint8.tif` – Input satellite hillshade 
- `prediction.tif` – Binary mask output (saved by code)
- `Upper_Willow_Creek_Roads_Buffer_3_Mask_1.tif` – Ground truth for MCC evaluation

### 🚀 How to Run Challenge 1
```bash
# Run segmentation prediction and save output
python3 segment_roads.py --input Upper_Willow_Creek_BareEarth_Hillshade.tif --output prediction.tif --checkpoint model.pth
```

### 📏 Evaluate Your Output
```python
from sklearn.metrics import matthews_corrcoef
import rasterio
import numpy as np

# Load prediction and ground truth masks
with rasterio.open("prediction.tif") as pred_sample:
    pred = pred_sample.read(1)

with rasterio.open("Upper_Willow_Creek_Roads_Buffer_3_Mask_1.tif") as true_sample:
    gt = true_sample.read(1)

# Ensure shape match
min_height = min(pred.shape[0], gt.shape[0])
min_width = min(pred.shape[1], gt.shape[1])
pred = pred[:min_height, :min_width]
gt = gt[:min_height, :min_width]

# Compute MCC score
score = matthews_corrcoef(gt.flatten(), pred.flatten())
print("✅ MCC Score:", score)
```

---

![image](https://github.com/user-attachments/assets/771d2d23-d8a0-427d-bbdf-906405b504c2)

## 🚗 Challenge 2: Bidirectional A* Pathfinding on Real Road Networks with F/G/H Simulation

This project implements a powerful Bidirectional A* pathfinding algorithm with animated search visualization and shapefile-based real-world road support. It includes dynamic F, G, H cost visualization and missing intersection recovery.

### 🌟 Features
- Bidirectional A* (forward & backward search)
- Dead-end fallback aware
- Real-time matplotlib simulation
- Dynamic F, G, H cost updates on-screen
- Auto-detect & insert missing intersections
- Custom and random start-goal support

### ⏱️ Runtime Complexity
- Bidirectional A*: O(b^(d/2)) (better than A*)
- Graph intersection detection: O(n^2) (acceptable for small datasets)

### 🧠 How It Works
- Uses GeoPandas and NetworkX to convert shapefiles to graphs
- All intersections detected and nodes added dynamically
- Bidirectional A* runs from both ends and stops at frontier meet
- Visual feedback with `plt.pause()` + shortest path animation

---

## 💻 How to Run Challenge 2

### 🔧 Install Required Libraries
```bash
pip install geopandas networkx shapely matplotlib
```

### ▶️ Run the Code
```bash
python3 hackathonProb.py
```

### 📂 Files
- `main - Prompt 2 - Luke.py` – Main file to launch the pathfinding
- `robot_class - Prompt 2 - Luke.py` – Robot class with navigation logic
- `edge - - Prompt 2 - Luke.py` – Graph edge definitions
- `node - Prompt 2 - Luke .py` – Node structure with cost calculations
- `South_Clear_Creek_Roads.shp` – Real road network shapefile
- `prompt2 solution.py` – Alternate entry-point or merged script

### 📍 Sample Coordinate Setup
```python
fixed_start = (x1, y1)
fixed_goal = (x2, y2)

start = get_nearest_node(G, fixed_start)
goal = get_nearest_node(G, fixed_goal)
```

---

## 🔗 Prompt Connection

Challenge 1 outputs a binary mask of detected roads from hillshade images. This mask can be vectorized into a shapefile which Challenge 2 then consumes to perform real-time A* pathfinding. This mirrors a real-world robotic workflow from terrain understanding to autonomous navigation.

Breakdown:

1. Drone captures LIDAR

2. LIDAR → DEM / hillshade image

3. Image → ML segmentation model → binary road mask (raster)

4. Raster → Vector conversion (e.g., .shp, GeoJSON)

5. Vector → Graph → use networkx for pathfinding

6. Visualize or simulate rover movement


