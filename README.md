# auto-path

## What if 

1. Road routes were recommended automatically?
2. Road bridges, viaducts, construction was planned ahead?

As of now, only **point 1** was covered

Let's see what tools we can build to facilitate this.


## API structure

The main endpoints.

### /terrain

Generate terrain with a given configuration and cache them using Redis.

The algorithms are programmed using celery.

#### POST /

```
payload:
{

}
```


### /planner

All actions regarding planning algorithm run on a specific terrain, including 
 timing, results and measuring the methods.

### /figures/mdpi-eng

Generates updated figures for ...

Will download them as a zip archive.

#### GET /

ERROR: should stop after a given timeout..