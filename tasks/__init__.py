from app import app
from utils import fibn_generator
# import tasks
import tasks_inner

@app.task(name="etc")
def long_task(*args, **kwargs):
    import time
    time.sleep(15)
    return 100

@app.task(name="fib_sum")
def fib_sum(*args, **kwargs):
    return tasks_inner.fib_sum(*args, **kwargs)


