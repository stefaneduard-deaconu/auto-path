from typing import Union, Optional
from celery import Celery
import redis
import dataclasses
from pymongo.database import Database
from cache import ExperimentCache, ExperimentField, HaCache, HagCache, HagbCache, HaghCache, OptionalExperimentField, TerrainCache


from main import Experiment


@dataclasses.dataclass
class Client:
    celery: Celery
    cache: redis.Redis
    db: Database

    def _get(self,
             collection: str,
             key: dict) \
            -> OptionalExperimentField:
        ans: OptionalExperimentField = self.db[collection] \
            .find_one(key,
                      {"_id": False})
        if not ans:
            ans = None
        return ans

    def set_terrain(self,
                    cfg_key: str,
                    value: TerrainCache):
        # TODO Ed, what to return?
        old = self.get_terrain(cfg_key=cfg_key)
        if old:
            # update:
            assert old['cfg_key'] == cfg_key, \
                "BAD FK"
            self.db['terrain'].update_one({'key': cfg_key},
                                          {'$set': value})  # TODO check values
        else:
            # create
            value['cfg_key'] = cfg_key
            self.db['terrain'].insert_one(value)\


    def get_terrain(self, cfg_key: str) -> Optional[TerrainCache]:
        return self._get('terrain', {"cfg_key": cfg_key})

    # # HA-->Terrain is a M:1 relationship
    # def get_ha(self, ha_key: str) -> Optional[HaCache]:
    #     # TODO Ed, how to implement? By using _id or sth else?
    #     return self._get('ha',
    #                      {"ha_key": ha_key})

    def get_ha_all(self, cfg_key: str) -> Optional[HaCache]:
        # TODO pay attention that 'ha' collection should keep both ha_key and cfg_key
        return self._get('ha',
                         {"cfg_key": cfg_key})

    # HAG->HA is a 1:1 relationship
    def get_hag(self, ha_key: str) -> Optional[HagCache]:
        return self._get('hag',
                         {"ha_key": ha_key})

    # HAGh->HA is a 1:1 relationship (if the heuristic is kept)
    def get_hag_h(self, ha_key: str) -> Optional[HaghCache]:
        return self._get('hagh',
                         {"ha_key": ha_key})

    # HAGb->HA is a 1:1 relationship (if the heuristic is kept)
    def get_hag_b(self, ha_key: str) -> Optional[HagbCache]:
        return self._get('hagb',
                         {"ha_key": ha_key})

    def get_experiment(self, config_key: str) -> ExperimentCache:
        # fetch terrain, ha, hag etc data
        # TODO Ed, fill in if more data is needed
        terrain_json = self.get_terrain(config_key)
        ha_json = self.get_ha_all(config_key)
        hag_json = self.get_hag(config_key)
        hag_h_json = self.get_hag_h(config_key)
        hag_b_json = self.get_hag_b(config_key)

        return {
            'terrain': terrain_json,
            'ha': ha_json,
            'hag': hag_json,
            'hag_b': hag_b_json,
            'hag_h': hag_h_json
        }
