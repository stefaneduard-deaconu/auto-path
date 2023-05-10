
from main import TerrainGeneratorConfig
# helper builtins
import itertools


def generate_configs_grid_based(seeds: list[int],
                                GRID_SIZES: list[tuple[int, int]],
                                scaling_arguments: list[tuple[int, int]],
                                height_intervals: list[tuple[int, int]],
                                height_deltas: list[int]) \
        -> list[TerrainGeneratorConfig]:
    configs = []
    for config_args in itertools.product(seeds,
                                         GRID_SIZES,
                                         scaling_arguments,
                                         height_intervals,
                                         height_deltas):
        gs, sa = config_args[1], config_args[2]
        # possible error: scaling_argument doesn't divide GRID_SIZE
        if gs[0] % sa[0] != 0:
            continue
        if gs[1] % sa[1] != 0:
            continue
        
        cfg = TerrainGeneratorConfig(*config_args)
        configs.append(cfg)
        
    return configs
