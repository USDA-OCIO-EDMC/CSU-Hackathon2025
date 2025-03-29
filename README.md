# LiDAR - Prompt 1
# RoverChallenge _ Prompt 2


# 🛣️ Bidirectional A* Pathfinding on Real Road Networks with Intersections & F/G/H Simulation

> 🚀 Hackathon Project - March 29, 2025  
> ✍️ Author: Vishnu Pradhaban J

This project implements a powerful Bidirectional A* pathfinding algorithm with animated search visualization and support for real-world road network data from shapefiles. The algorithm is enhanced to detect and insert missing intersection nodes in the graph and displays real-time F (total), G (cost), and H (heuristic) values on the screen.

## 🔍 Features
- Bidirectional A* pathfinding (forward and backward search from start and goal)
- Fallback-aware search: continues exploring if one direction hits a dead-end
- Live simulation of the search process using matplotlib
- G, H, F scores displayed dynamically on-screen
- Detects and adds intersection nodes where roads cross without a node
- Supports both fixed and random start-goal pairs
- Visualizes road network and search steps side by side

## ⏱️ Runtime & Time Complexity

### Time Complexity
- **Bidirectional A\*** typically reduces the search space compared to standard A*:
  - Worst-case complexity: **O(b^(d/2))**, where:
    - `b` = branching factor (average neighbors per node)
    - `d` = depth (number of steps in the shortest path)
- The graph construction and intersection insertion takes:
  - **O(n²)** in the worst case for all edge pair comparisons (`n` = number of edges)
  - But is acceptable on small road networks like the Clear Creek area

### Runtime Notes
- Visualization introduces artificial delay using `plt.pause()`, which can be tuned
- Pathfinding on medium graphs (500–3000 nodes) typically completes in **~1–5 seconds**
- Real-time plotting of each node during exploration helps with educational demos but can be disabled for benchmarking

## 🧪 How It Works
- Loads a shapefile using GeoPandas and builds a graph using NetworkX
- Each LineString segment becomes multiple edges; points become nodes
- All segment pairs are checked for intersections; if found, new nodes are added and original edges are split
- Bidirectional A* runs from start and goal; whenever the two frontiers meet, path is reconstructed
- At each step, explored nodes are plotted and F, G, H values are shown on screen
- The final shortest path is drawn over the road map

## 🖥️ How to Run
1. Make sure you have the required libraries:
```bash
pip install geopandas networkx shapely matplotlib




Run the command to run this code:
python3 hackathonProb.py

hackathonProb.py               # Main project script
South_Clear_Creek_Roads.shp   # Shapefile data
README.md                     # This file



	Fixed coordinates using:

fixed_start = (x1, y1)
fixed_goal = (x2, y2)
start = get_nearest_node(G, fixed_start)
goal = get_nearest_node(G, fixed_goal)
