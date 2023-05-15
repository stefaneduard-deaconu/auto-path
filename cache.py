from typing import TypedDict, Union, Optional


class ConfigCache(TypedDict):
    key: str
    seed: int
    GRID_SIZE: tuple[int, int]
    scaling_argument: tuple[int, int]
    height_interval: tuple[int, int]


class TerrainCache(TypedDict):
    cfg_key: str
    surf: list[list[float]]
    min_height: float
    max_height: float



class HaCache(TypedDict):
    trn_key: str
    height_delta: float
    surf_h: list[list[float]]
    buckets: list[float]


class HagCache(TypedDict):
    ha_key: str
    pass  # TODO Ed,


class HaghCache(TypedDict):
    ha_key: str
    pass   # TODO Ed,


class HagbCache(TypedDict):
    ha_key: str
    pass  # TODO Ed,


ExperimentField = Union[TerrainCache,
                        HaCache,
                        HagCache,
                        HaghCache,
                        HagbCache]
OptionalExperimentField = Optional[ExperimentField]

def init_args(cache: ExperimentField,
              ignore_keys: tuple[str] = ('cfg_key', 'key')):
    return {k:v for k,v in cache.items()
            if k not in ignore_keys}




class ExperimentCache(TypedDict):
    config: ConfigCache
    terrain: Optional[TerrainCache]
    ha: Optional[HaCache]
    hag: Optional[HagCache]
    hag_h: Optional[HaghCache]
    hag_b: Optional[HagbCache]
