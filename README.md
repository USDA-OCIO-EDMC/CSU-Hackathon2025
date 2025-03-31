# 🛰️ LiDAR - Prompt 1 & RoverChallenge - Prompt 2

## 📍 Challenge 1: Road Segmentation from LiDAR Hillshade

This module tackles the task of detecting roads from hillshade LiDAR data using a custom-trained U-Net-based segmentation model.

### 👩‍💻 Team
- Vishnu Jawahar - [vishnu.lvj@gmail.com](mailto:vishnu.lvj@gmail.com)
- Luke Burrough - [luke.burrough@colostate.edu](mailto:luke.burrough@colostate.edu)
- Oshin Tiwari - [oshintiwari0928@gmail.com](mailto:oshintiwari0928@gmail.com)
- Samuel Chamberlain - [Samuel.Chamberlain@rams.colostate.edu](mailto:Samuel.Chamberlain@rams.colostate.edu)

### 📌 Key Features
- ✅ U-Net model for binary segmentation of roads from LiDAR hillshade input
- ✅ Inference function that slides over large images with overlap and stitching
- ✅ Threshold tuning and sigmoid sharpening for better mask quality
- ✅ Final prediction exported as GeoTIFF for vectorization
- ✅ MCC (Matthews Correlation Coefficient) based evaluation pipeline

### 🛠️ Tools & Libraries
Python, PyTorch, Rasterio, NumPy, Matplotlib, Scikit-learn

### 📂 File Outputs
- `prediction.tif` — Final predicted binary road mask
- `.shp` — Vectorized shapefile of detected roads (optional)
- Score output — Evaluation against ground truth using MCC

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

### 📁 Files Used
```
hackathonProb.py                  # Main script
South_Clear_Creek_Roads.shp      # Shapefile road network
README.md                         # This documentation
```

### 📍 Sample Coordinate Setup
```python
fixed_start = (x1, y1)
fixed_goal = (x2, y2)

start = get_nearest_node(G, fixed_start)
goal = get_nearest_node(G, fixed_goal)
```

---

## 📌 Folder Structure

```
├── segment_roads.py             # Challenge 1 segmentation
├── hackathonProb.py             # Challenge 2 pathfinding
├── model.pth                    # Trained segmentation model
├── prediction.tif               # Output mask
├── South_Clear_Creek_Roads.shp  # Road shapefile
├── Upper_Willow_Creek_*.tif     # Input hillshade or DEM tiles
├── README.md                    # This file
```

---

## 🔗 Prompt Connection

Challenge 1 outputs a binary mask of detected roads from hillshade images. This mask can be vectorized into a shapefile which Challenge 2 then consumes to perform real-time A* pathfinding. This mirrors a real-world robotic workflow from terrain understanding to autonomous navigation.

