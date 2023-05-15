from tests_data import config

from tasks import task_terrain_from_config, task_HA_from_config, task_HAG_from_config

# TODO Ed, height doesn't seem right
# TODO Ed, save the data to a database
res = task_terrain_from_config(config)
print(res)
res2 = task_terrain_from_config(config)
assert res2 == res
