
from cache import HaCache, HagCache, TerrainCache
from client import Client
from main import Experiment, TerrainGeneratorConfig
from redis import Redis


def restore_experiment(config: TerrainGeneratorConfig,
                       client: Client) -> Experiment:
    # get the terrain data:
    # TODO Ed, create methods inside experiment in the future
    experiment_merged_json = client.get_experiment(config.key)
    return Experiment.from_cached(data=experiment_merged_json,
                                  config=config)

def update_experiment(config: TerrainGeneratorConfig,
                      client: Client,
                      e: Experiment):
    
    # save to DB: TODO Ed, make an experiment update function
    # TODO Ed, make a function client.update_experiment() to update whole experiment? NO, it would use Experiment class
    #  so we rather make a local function inside tasks inner !!!
    client.set_terrain(cfg_key=config.key,
                       value=e.json_terrain)

# TODO Ed,
#      MAYBE, we can cache the experiment inside REDIS?
#      OR maybe a function to run experiment from the start to a given stage, using caching or just the Mongo database

def task_inner_terrain_from_config(config: TerrainGeneratorConfig,
                                   client: Client) \
        -> TerrainCache:
    e = restore_experiment(config, client)
    breakpoint()
    if not e.area:
        # generate terrain
        e.generate_terrain()
    # update the experiment data, TODO Ed, may run excessive updates, should only run for the updated data, we could mark using a dictionary...
    update_experiment(config, client, e)
    return e.json_terrain


def task_inner_HA_from_config(config: TerrainGeneratorConfig,
                              client: Client) \
        -> HaCache:
    e = restore_experiment(config, client)
    breakpoint()
    if not e.area:
        # generate terrain
        e.generate_terrain()
    if not e.area_sections:
        # generate height areas
        e.generate_height_areas()  # TODO Ed
    return e.json_height_areas  # TODO Ed


def task_inner_HAG_from_config(config: TerrainGeneratorConfig,
                               client: Client) \
        -> HagCache:
    e = restore_experiment(config, client)
    breakpoint()
    if not e.area:
        # generate terrain
        e.generate_terrain()
    if not e.area_sections:
        # generate terrain
        e.generate_height_areas()  # TODO Ed
    # TODO use the experiment  to generate ha
    if not e.hag:
        e.generate_hag()
    return e.json_hag
