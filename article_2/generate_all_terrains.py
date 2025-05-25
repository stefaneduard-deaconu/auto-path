from main import *
import matplotlib
from data.configs import fast_configs, slow_configs

matplotlib.use('TkAgg')

if __name__ == '__main__':
    # 1. Run all fast configs
    for config in fast_configs:
        # TODO Ed, set the square size
        with timer("generate fast experiment"):
            e = Experiment(config=config)
            e.generate(cache=True)
    print('**************************',
          '*** fast terrains DONE ***',
          '**************************', sep='\n')
    # 2. Run all slow configs
    for config in slow_configs:
        # TODO Ed, set the square size
        with timer("generate slow experiment"):
            e = Experiment(config=config)
            e.generate(cache=True)
    print('**************************',
          '*** SLOW terrains DONE ***',
          '**************************', sep='\n')
    # 3. run Dijkstra
    for config in fast_configs:
        # TODO Ed, set the square size
        with timer("path planning based on slow experiment"):
            e = Experiment(config=config)
            e.generate(cache=True)
            start, target = (5, 5), (config.GRID_SIZE[0] - 5, config.GRID_SIZE[1] - 5)
            e.reset_objective(start, target)
            paths = e.test_dijkstra_variants(cache=True,noshow=True, only_compute_based_on='height')
    print('**************************',
          '*** fast planning DONE ***',
          '**************************', sep='\n')
    for config in slow_configs:
        # TODO Ed, set the square size
        with timer("path planning based on fast experiment"):
            e = Experiment(config=config)
            e.generate(cache=True)
            start, target = (5, 5), (config.GRID_SIZE[0] - 5, config.GRID_SIZE[1] - 5)
            e.reset_objective(start, target)
            paths = e.test_dijkstra_variants(cache=True, noshow=True, only_compute_based_on='height')
    print('**************************',
          '*** SLOW planning DONE ***',
          '**************************', sep='\n')
