Trail Navigation Project

Overview
This project uses A* pathfinding and machine learning (LSTM) to find the shortest path across a binary mask of trails or paths. The pathfinding results are visualized using Pygame and TensorFlow.

Requirements
Python 3.x

pygame, tensorflow, opencv, numpy, tifffile, Pillow

To install dependencies:
pip install pygame tensorflow opencv-python numpy 

you may also need to:
pip install os math heapq time

Prepare the TIFF File:

The program needs a binary mask image of trails/paths in TIFF format (.tif) with 6380x8000 pixels or smaller.

The paths should be white pixels and the background black.

Run the Script:

python model.py

Enter TIFF File Path:

When prompted, enter the full path to the .tif file (e.g., /path/to/file.tif).

If the file is invalid or not found, the program will generate a random binary mask.

Output:

The program will first show the A* pathfinding results.

Then, the machine learning model will display its solution.

Approaches
A* Pathfinding:

The A* algorithm finds the shortest path in the binary mask image and is visualized using Pygame.

Machine Learning Model:

The model (LSTM) predicts the path based on learned sequences from the A* solution.

Fallback to Random Trails:

If the input file is invalid, a random binary mask is generated, and both pathfinding approaches are tested on it.

Final Approach
A* algorithm finds the initial path.

The LSTM model refines the pathfinding solution.

Both solutions are displayed for comparison.

Output: Both A* and ML pathfinding results are shown in pop-up windows.