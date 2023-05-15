from app import client
from cache import ConfigCache, HaCache, HagCache, TerrainCache
from main import TerrainGeneratorConfig

# import tasks inner code
from tasks_inner import task_inner_terrain_from_config, \
                        task_inner_HA_from_config, \
                        task_inner_HAG_from_config

from celery.contrib import rdb  # TODO Ed, remove on merge
import json

# TODO Ed:
#  roles of tasks: deserialize input data, send to inner tasks,
#                  and return json data as objects
#  roles of inner tasks: try grabbbing data from DB, run computations,
#                        and update the DB with the new data

@client.celery.task(serializer='json')
def task_terrain_from_config(config_dict: ConfigCache) \
        -> TerrainCache:  # TODO Ed, to annotate the config, I must refactor the config class to be importable
    config: TerrainGeneratorConfig = TerrainGeneratorConfig(**config_dict)
    return task_inner_terrain_from_config(config,
                                          client=client)

@client.celery.task(serializer='json')
def task_HA_from_config(config_dict: ConfigCache) \
        -> HaCache:
    config: TerrainGeneratorConfig = TerrainGeneratorConfig(**config_dict)
    return task_inner_HA_from_config(config, 
                                     client=client)

# TODO Ed, also cache the big experiment?

@client.celery.task(serializer='json')
def task_HAG_from_config(config_dict: ConfigCache) \
        -> HagCache:
    config: TerrainGeneratorConfig = TerrainGeneratorConfig(**config_dict)
    return task_inner_HAG_from_config(config, 
                                      client=client)
    