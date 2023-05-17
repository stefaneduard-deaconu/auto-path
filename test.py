from tests_data import config

from tasks import task_terrain_from_config, task_HA_from_config, task_HAG_from_config

# TODO Ed, height doesn't seem right
# TODO Ed, save the data to a database
res = task_terrain_from_config(config)
res2 = task_terrain_from_config(config)
assert res2 == res

ha1 = task_HA_from_config(config)
ha2 = task_HA_from_config(config)
assert ha1 == ha2

hag1 = task_HAG_from_config(config)
hag2 = task_HAG_from_config(config)
assert hag1 == hag2
