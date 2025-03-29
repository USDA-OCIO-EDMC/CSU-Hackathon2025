import heapq
import random
import numpy as np

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
    Simulates a player on the game map.
    Codes:
      0 = background, 1 = path, 2 = wall, 3 = reward.
    If start or reward positions are not provided, they are chosen at random from non-wall cells.
    Returns the optimal path as a list of (row, col) tuples.
    """
    rows, cols = game_map.shape

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
