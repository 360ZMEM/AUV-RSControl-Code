import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
terrain = np.load('terrain.npy')

constrain = lambda x_min, x, x_max: x if (x_min <= x <= x_max) else (x_min if x < x_min else x_max)

def get_terrain_height(x_max, y_max, x_now, y_now): 
    x_grid, y_grid = terrain.shape
    x_idx = int((x_grid - 1) * constrain(0, x_now / x_max, 1))
    y_idx = int((y_grid - 1) * constrain(0, y_now / y_max, 1))
    return - terrain[x_idx, y_idx]