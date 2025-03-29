import math
import heapq


class Cell:
    def __init__(self):
        self.parent_i = 0
        self.parent_j = 0
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0


ROW = 10460
COL = 13115


class Algorithm:
    @staticmethod
    def is_valid(row, col):
        return (row >= 0) and (row < ROW) and (col >= 0) and (col < COL)

    @staticmethod
    def is_unblocked(grid, row, col):
        return grid[row][col] == 1

    @staticmethod
    def is_destination(row, col, dest):
        return row == dest[0] and col == dest[1]

    @staticmethod
    def calculate_h_value(row, col, dest):
        return math.sqrt((row - dest[0]) ** 2 + (col - dest[1]) ** 2)

    @staticmethod
    def trace_path(cell_details, dest, grid):
        print("The Path is:")
        path = []
        row, col = dest

        # Backtrack to find the path
        while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
            path.append((row, col))
            temp_row = cell_details[row][col].parent_i
            temp_col = cell_details[row][col].parent_j
            row = temp_row
            col = temp_col

        path.append((row, col))  # Add the source cell
        path.reverse()

        path_grid = [[0 for _ in range(COL)] for _ in range(ROW)]

        for (r, c) in path:
            path_grid[r][c] = 1  # Mark the path with 1 or any other value

        return path_grid

    @staticmethod
    def a_star_search(grid, src, dest):
        if not Algorithm.is_valid(src[0], src[1]) or not Algorithm.is_valid(dest[0], dest[1]):
            print("Source or destination is invalid")
            return None

        if not Algorithm.is_unblocked(grid, src[0], src[1]) or not Algorithm.is_unblocked(grid, dest[0], dest[1]):
            print("Source or the destination is blocked")
            return None

        if Algorithm.is_destination(src[0], src[1], dest):
            print("We are already at the destination")
            # Return path grid with the source
            path_grid = [[0 for _ in range(COL)] for _ in range(ROW)]
            path_grid[src[0]][src[1]] = 1
            return path_grid  # Return a grid with just the source as the path

        closed_list = [[False for _ in range(COL)] for _ in range(ROW)]
        cell_details = [[Cell() for _ in range(COL)] for _ in range(ROW)]

        i, j = src
        cell_details[i][j].f = 0
        cell_details[i][j].g = 0
        cell_details[i][j].h = 0
        cell_details[i][j].parent_i = i
        cell_details[i][j].parent_j = j

        open_list = []
        heapq.heappush(open_list, (0.0, i, j))

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]

        while len(open_list) > 0:
            p = heapq.heappop(open_list)
            i, j = p[1], p[2]
            closed_list[i][j] = True

            for dir in directions:
                new_i, new_j = i + dir[0], j + dir[1]

                if Algorithm.is_valid(new_i, new_j) and Algorithm.is_unblocked(grid, new_i, new_j) and not closed_list[new_i][new_j]:
                    if Algorithm.is_destination(new_i, new_j, dest):
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j
                        print("The destination cell is found")
                        # Trace and return the populated path grid
                        return Algorithm.trace_path(cell_details, dest, grid)

                    g_new = cell_details[i][j].g + 1.0
                    h_new = Algorithm.calculate_h_value(new_i, new_j, dest)
                    f_new = g_new + h_new

                    if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        cell_details[new_i][new_j].f = f_new
                        cell_details[new_i][new_j].g = g_new
                        cell_details[new_i][new_j].h = h_new
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j

        print("Failed to find the destination cell")
        return None

    @staticmethod
    def startA(grid_tert):
        dest = [10459, 13114]
        src = [0, 0]
        path = Algorithm.a_star_search(grid_tert, src, dest)
        return path