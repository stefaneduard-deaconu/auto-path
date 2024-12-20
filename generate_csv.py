from generate_article_figures import save_np_to_csv, smooth_path
from main import *
from itertools import product

import math

import matplotlib

matplotlib.use('TkAgg')

SCALE = 10    

if __name__ == "__main__":
    # 1. Choose an area and plot it
    
    # config = TerrainGeneratorConfig(
    #     seed=0,
    #     GRID_SIZE=(100, 100),
    #     scaling_argument=(4, 4),
    #     height_interval=(100, 120),
    #     height_delta=3
    # )
    config = TerrainGeneratorConfig(
        seed=7, GRID_SIZE=(60, 60),
        scaling_argument=(2, 2),
        height_interval=(320, 336),
        height_delta=2
    )
    
    start, target = (
        (15, 15),
        (config.GRID_SIZE[0] - 15, config.GRID_SIZE[1] - 15)
    )
    
    # TODO Ed, set the square size
    with timer("generate experiment"):
        e = Experiment(config=config)
        e.generate(cache=True)
    
    # e.area_sections.orig_area.surf contains the 3D points of the terrain
        
    terrain_matrix = e.area_sections.orig_area.surf
    n, m = terrain_matrix.shape
    terrain_pts = np.array([
        (ln, cl, terrain_matrix[ln,cl]) 
        for ln, cl in product(range(n), range(m))
    ])

    # set start, target, and then generate road and the smooth road
    e.area_sections.orig_area.start, e.area_sections.orig_area.target = (start, target)
    path = e.test_dijkstra_variants(cache=True, noshow=True)['height']
    smooth_path = smooth_path(path)
    smooth_path_3d = np.array(e.area_sections.interpolate_path_height(path))
    
    save_np_to_csv(smooth_path_3d, 'terrain_path.csv')
    save_np_to_csv(terrain_pts, 'terrain_area.csv')
