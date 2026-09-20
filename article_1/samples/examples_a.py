import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from areas.area import segment_length
from areas.utils import set_axes_equal, create_subplots
from areas.utils.interpolate import remove_bad_points, path_length, interpolate_2d_path_as_is, direction
from main import *

import math

import matplotlib

matplotlib.use('TkAgg')

SCALE = 10    

if __name__ == "__main__":
    # 1. Choose an area and plot it
    config = TerrainGeneratorConfig(
        seed=0,
        GRID_SIZE=(100, 100),
        scaling_argument=(4, 4),
        height_interval=(100, 120),
        height_delta=3
    )
    # TODO Ed, set the square size
    with timer("generate experiment"):
        e = Experiment(config=config)
        e.generate(cache=True)
        
    # e.area.pts3d contains the 3D points of the terrain
    print(e.area.pts3d)
