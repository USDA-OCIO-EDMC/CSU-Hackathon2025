Team Members:

    Name: Ethan Luu
    Email: eluu@colostate.edu
    Github username: eluu02

    Name: Miguel Estrada
    Email: C837279570@colostate.edu
    Github username: Mateo-mme

    Name: Ethan Arroyo
    Email: c835266433@colostate.edu
    Github username: ethanarroyo68

Runtime:
Approx. 27 seconds

Approaches:

1. Assumption that all nodes were connected, simply used A* to determine shortest path
2. Account for lack of connected graph, only able to measure distances along individual paths
3. Optimization; only detect intersections that the rover encountered as it moved along a path. Used a moving bounded box.

Final Approach:

Addition of intersections before running, use of A* to create a path along roads for the shortest path.
