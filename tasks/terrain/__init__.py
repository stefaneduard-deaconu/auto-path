from app import app
from main import TerrainGeneratorConfig

# import tasks inner code
from tasks_inner import task_inner_terrain_from_config

# from celery.contrib import rdb  # TODO Ed, remove on merge


@app.task(serializer='json')
def task_terrain_from_config(config_dict: dict):  # TODO Ed, to annotate the config, I must refactor the config class to be importable
    config: TerrainGeneratorConfig = TerrainGeneratorConfig(**config_dict)
    return task_inner_terrain_from_config(config)