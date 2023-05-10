
from main import Experiment, TerrainGeneratorConfig

def task_inner_terrain_from_config(config: TerrainGeneratorConfig):
    e = Experiment(config=config)
    e.generate(cache=True)  # will generate terrain
    return e.json_terrain
    # could pass the save argument to task, to toggle whether only numpy/graph data should be created