from app import app
from main import Experiment, TerrainGeneratorConfig

from celery.contrib import rdb

@app.task(serializer='json')
def task_terrain_from_config(config_dict: dict):  # TODO Ed, to annotate the config, I must refactor the config class to be importable
    config: TerrainGeneratorConfig = TerrainGeneratorConfig(**config_dict)
    e = Experiment(config=config)
    e.generate(cache=True)
    return e.test_dijkstra_variants(noshow=True, save=True)  # could pass the save argument to task, to toggle whether only numpy/graph data should be created