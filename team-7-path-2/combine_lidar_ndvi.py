import rasterio
import numpy as np
import math


ndvi_data = np.load('./ndvi_points_no_null.npy')

lidar_data = np.load('./lidar_points_2.npy')

print(f'lidar shape {lidar_data.shape}')
print(f'ndvi shape {ndvi_data.shape}')


joined_data = np.zeros((len(lidar_data), len(lidar_data[0])))


def get_slope(start, end):
    delta_y = abs(end - start)
    return math.atan(delta_y / 1)


def get_scaler(slope):
    if slope < .2:
        return 1.0
    elif slope >= .2 and slope < .6:
        return 1.25
    elif slope >= .6 and slope < .8:
        return 1.5
    return 2


def calculate_final_value(ndvi, slope_scaler):
    if ndvi < 0 or ndvi >= .5:
        return 100
    return ndvi * slope_scaler


actions = [[0, 1], [0, -1], [-1, 0], [1, 0],
           [1, 1], [1, -1], [-1, 1], [-1, -1]]
# have confirmed previously that they match
print("joining data")
for j in range(lidar_data.shape[0]):
    for i in range(j, lidar_data.shape[1]):
        start_cell = lidar_data[j, i]
        try:
            for action in actions:
                end_i = i + action[0]
                end_j = j + action[1]
                end_cell = lidar_data[end_j, end_i]
                slope = get_slope(start_cell, end_cell)
                slope_scaler = get_scaler(slope)
                ndvi_value = ndvi_data[j, i]
                updated_value = calculate_final_value(ndvi_value, slope_scaler)
                joined_data[j, i] = updated_value
                joined_data[end_j, end_i] = updated_value

        except IndexError:
            continue
print(joined_data.shape)
np.save('joined_data_no_null.npy', joined_data)
