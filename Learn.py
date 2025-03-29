import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from robot_class import Robot
from TertiaryCostMap import CostMap
from A import Algorithm
from Visualizer import visualize
import numpy as np
import torch.nn as nn
from OptimalPathFinder import OptimalPathFinder

# Train a the Optimal Path Finder model over various iterations


class Learn:
    def train():

        runs = 10
        success = 0
        # our model is Optimal Path Finder
        finder = OptimalPathfinder()

        # run multiple times to train
        for x in range(runs):
            path = finder.run()
            if path_valid(path):
                success += 1

        accuracy = (success/runs) * 100
        
        #print(f"Path's accuracy over {runs}: {accuracy:.2f}%")

    def path_valid(path):
        # check to make sure the path does not contain any 1's
        # check to make sure the path reaches the destination and begins from the source


