"""
How to run: start the 4 celery workers, after which you run thie file in another process
"""

from pprint import pprint
import time
from tasks import task_terrain_from_config
from utils import generate_configs_grid_based

# generate close to 3*2*4*1*4 = 96 configurations
configs = generate_configs_grid_based(
    seeds=[0, 1, 2],
    GRID_SIZES=[(60, 60), (100, 100)],
    scaling_arguments=[(1, 1), (2, 2), (3, 3), (4, 4)],
    height_intervals=[(100, 200)],
    height_deltas=[2, 3, 4, 5]
)
configs = generate_configs_grid_based(
    seeds=[11],
    GRID_SIZES=[(60, 60)],
    scaling_arguments=[(2, 2)],
    height_intervals=[(100, 200)],
    height_deltas=[5]
)

# run a task for each configuration
async_results = [task_terrain_from_config.delay(cfg.to_json)
                 for cfg in configs]

# print new results after each 2 seconds
tasks_success = []
tasks_failure = []
while async_results:
    time.sleep(2)
    async_results_done = [res for res in async_results
                          if res.status in ['SUCCESS', 'FAILURE']]
    for res in async_results_done:
        print(f'Task {res.id}: {res.status} -- ')
        if res.status == 'SUCCESS':
            # print(res.result)  # TODO Ed, too verbose
            tasks_success.append(res)
        elif res.status == 'FAILURE':
            tasks_failure.append(res)
    async_results = [res for res in async_results
                     if res.status == 'PENDING']
print("---",
      "All tasks ended",
      "---", sep='\n')
print("SUCCESSES")
pprint(tasks_success)
print("FAILURES")
pprint(tasks_failure)
