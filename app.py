import redis
import celery
import pymongo
from pymongo.database import Database

# Chaining
from celery import Celery, chain
from client import Client

# localhost:6379/0 is used for celery, and
# localhost:6379/100 is used for caching experiment data

app = Celery('planner', 
             broker='redis://localhost',
             backend="redis://localhost/0")
redis_cache = redis.Redis(db=1)
db: Database = pymongo.MongoClient("mongodb://localhost:27017/")['auto_path']

client = Client(celery=app, 
                cache=redis_cache, 
                db=db)

from tasks import *