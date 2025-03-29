def get_neighboring_block_values(grid, center_x, center_y, detection_distance):
    """
    Retrieves the neighboring block values in a grid within a certain detection distance.

    Args:
    grid (list): A 2D list representing the grid.
    center_x (int): The x-coordinate of the center block.
    center_y (int): The y-coordinate of the center block.
    detection_distance (int): The maximum distance the sensor can detect.

    Returns:
    list: A list of neighboring block values.
    """
    # Initialize an empty list to store the neighboring block values
    neighboring_values = []

    # Iterate over each possible distance
    for distance in range(1, detection_distance + 1):
        # Calculate the coordinates for the 8 directions
        for direction_x, direction_y in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            # Calculate the new x and y coordinates
            new_x, new_y = center_x + distance * direction_x, center_y + distance * direction_y

            # Check if the new coordinates are within the grid boundaries
            if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]):
                # Append the neighboring block value to the list
                neighboring_values.append(grid[new_x][new_y])

    return neighboring_values

# Example usage:
grid = [
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
    [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
    [33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0],
    [41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
    [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
    [57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0]
]

center_x, center_y = 3, 3
detection_distance = 2
neighboring_values = get_neighboring_block_values(grid, center_x, center_y, detection_distance)
print(neighboring_values)
