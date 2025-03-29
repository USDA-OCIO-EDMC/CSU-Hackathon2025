import heapq

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, start, end):
    current = end
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

def astar(grid, start, end):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == end:
            return reconstruct_path(came_from, start, end)

        for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            y, x = current[0] + dy, current[1] + dx
            if 0 <= y < len(grid) and 0 <= x < len(grid[0]) and grid[y][x] != 1:
                new_cost = cost_so_far[current] + 1

                if (y, x) not in cost_so_far or new_cost < cost_so_far[(y, x)]:
                    cost_so_far[(y, x)] = new_cost
                    priority = new_cost + heuristic((y, x), end)
                    heapq.heappush(open_list, (priority, (y, x)))
                    came_from[(y, x)] = current

    return None

grid = [
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

start = (0, 0)
end = (9, 9)

path = astar(grid, start, end)

if path:
    print("Shortest path found:")
    for point in path:
        print(point)
else:
    print("No path found")

# Print the grid with the shortest path
for y in range(len(grid)):
    for x in range(len(grid[0])):
        if (y, x) in path:
            print('*', end=' ')
        elif grid[y][x] == 1:
            print('#', end=' ')
        else:
            print('.', end=' ')
    print()

