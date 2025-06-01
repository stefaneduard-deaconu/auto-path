# generate 100 terrain configurations
from itertools import product

from main import TerrainGeneratorConfig
from typing import Iterable


def generate_configs(
        seeds: Iterable[int],
        grid_sizes: Iterable[int],
        scaling_arguments: Iterable[int],
        height_deltas: Iterable[int] = (1, 2, 3, 5),
) -> list[TerrainGeneratorConfig]:
    return [
        TerrainGeneratorConfig(
            seed=seed,
            GRID_SIZE=(grid_size, grid_size),
            scaling_argument=(scale_arg, scale_arg),
            height_interval=(100, 150),
            height_delta=height_delta
        )
        for seed, grid_size, scale_arg, height_delta in product(
            seeds,
            grid_sizes,
            scaling_arguments,
            height_deltas,
        )
        if grid_size % scale_arg == 0
    ]


over_100_seeds_for_algorithm1 = generate_configs(
    range(5, 105),
    [50],
    [2],
    [(3)]
)

fast_configs = generate_configs(
    range(3, 10),
    [20, 50, 80],
    range(2, 5)
)

slow_configs = generate_configs(
    range(3),
    [120, 150, 180],
    range(2, 5)
)

all_configs = [
    *fast_configs,
    *slow_configs,
]

article_config = TerrainGeneratorConfig(
    seed=0, GRID_SIZE=(100, 100),
    scaling_argument=(4, 4),
    height_interval=(100, 150),
    height_delta=3
)
