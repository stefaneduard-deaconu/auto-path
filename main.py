import contextlib
import dataclasses
import time
import itertools
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from areas.area_sections import Area, AreaSections
from areas.area import eucl, interpolate_using_bezier_curve

# if __name__ == '__main__':
#     # # section the area by height :)
#     # a_sections = AreaSections.from_area(area=a, height_delta=10)
#
#     # # plot heat map
#     # a_sections.plot_heatmap()
#     # a_sections.show()
#
#     # # plot all separate area
#     # a_sections.plot_areas()
#     # a_sections.show()
#
#     # # TODO Ed, plot candidate corridors
#     # for h_diff in range(0, 90, 10):  # TODO Ed, compute these resursively for faster results
#     #     a_sections.plot_grouped(max_hdiff=h_diff)
#     #     a_sections.show()
#     # # TODO Ed, compute the available areas better and gradually
#     # a_sections.plot_grouped(max_hdiff=80)
#
#     # TODO Ed P0, implement
#
#     # TODO Ed, implement functions for showcasing a chosen road, using the samples below:
#     # # line = interpolate_3d_line(area=a,
#     # #                            start=a_sections.start,
#     # #                            target=a_sections.target)
#     # line = interpolate_using_bezier_curve(area=a,
#     #                                       waypoints=[a.start,
#     #                                                  [340,  270],
#     #                                                  [250,  290],
#     #                                                  [100,  70],
#     #                                                  [5,    225],
#     #                                                  a.target])

#
#     # TODO Ed, task 1 - implement a function to 3d print the area around a point, using the Blues color map
#     # a.plot_vicinity(a.start, radius=50)
#     # a.plot_terrain_3d(axis='equal', use_tk=False)
#
#     # wg = a.generate_graph()
#     # path = wg.shortest_path((0,0), (10,3))
#     # print(wg.path_weight(path))
#     # a.plot_vicinity((5,5), 5, use_tk=True)

from functools import wraps
import time

from areas.graph import generate_hsections_graph, Coord
from areas.utils import create_3d_subplots, set_axes_equal
from areas.utils.interpolate import interpolate_2d_path


def timeit(func):
    """decorator for timing a function call"""

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def slow_method_using_full_graph():
    """section the area by height as efficiently as possible:"""

    a = Area.from_perlin_noise(seed=10, GRID_SIZE=(500, 500), min_height=200, max_height=300)
    a.generate_objective(seed=123)
    a.plot_terrain()
    a.show()

    @timeit
    def compute_area_sections(a: Area,
                              height_delta: float = 0,
                              max_hdiff: float = 0):
        # TODO Ed, task3 - implement better area sections computing
        a_sections = AreaSections.from_area(area=a, height_delta=16)
        a_sections.plot_grouped(max_hdiff=70)
        # plot heat map
        a_sections.plot_heatmap()
        a_sections.show()

    for hd in [10, 15, 20, 25]:
        compute_area_sections(a,
                              height_delta=hd,
                              max_hdiff=70)


# TODO Ed, create an context for function calls

@contextlib.contextmanager
def timer(codeblock_name: str):
    """
    From https://towardsdatascience.com/how-to-build-custom-context-managers-in-python-31727ffe96e1
    """
    # Start the timer
    start = time.time()
    # context breakdown
    yield
    # End the timer
    time_elapsed = time.time() - start
    # Tell the user how much time elapsed
    print(f'<<{codeblock_name}>> executed in {"%.4f" % time_elapsed} seconds.')


@dataclasses.dataclass(unsafe_hash=True)
class TerrainGeneratorConfig:
    seed: int
    GRID_SIZE: tuple[int, int]
    scaling_argument: tuple[int, int]
    height_interval: tuple[int, int]
    height_delta: int

    @property
    def for_area(self):
        return {k: v
                for k, v in self.__dict__.items()
                if k is not 'height_delta'}

    @property
    def for_area_sections(self):
        return {"height_delta": self.height_delta}


# TODO Ed, current task
#  1. Classify roads based on inclination
#    to do this, we need to interpolate them to a smooth curve..
#  2. Add errors such as ImpossibleSurfaceRoad (because the inclination is too high/low
#  3. And what about the ones that are dangerous, but not impossible?

class CacheFileNotFoundException(Exception):
    pass


class Experiment:
    def __init__(self,
                 config: TerrainGeneratorConfig):
        self.config = config
        self.area: Area = None
        self.area_sections: AreaSections = None

    def _generate_from_scratch(self):
        self.area = Area.from_perlin_noise(**self.config.for_area)
        self.area_sections = AreaSections.from_area(area=self.area,
                                                    **self.config.for_area_sections)

    def reset_objective(self, start: Coord, target: Coord):
        self.area_sections.update_objective(start, target)

    def generate(self,
                 cache: bool = True):
        try:
            if cache:
                with timer(f'from cache, config={self.config}'):
                    self.from_cache()
            else:
                raise CacheFileNotFoundException("create the cache this time")
        except Exception as e:
            if not isinstance(e, CacheFileNotFoundException):
                raise
            # not cached, meaning that we load
            with timer('NO CACHE => generate and cache'):
                self._generate_from_scratch()
                self.to_cache()

    @property
    def cache_fn(self):
        # TODO Ed, error from uncaching...
        try:
            import os
            os.mkdir('experiments')
        except:
            pass
        return f"experiments/experiment{self.config.__hash__()}.cache"

    def to_cache(self):
        def is_picklable(obj):
            import pickle
            try:
                pickle.dumps(obj)

            except pickle.PicklingError:
                return False
            return True

        fn = self.cache_fn
        try:
            import pickle as pkl
            # What to cache:
            cache_dict = deepcopy(self.__dict__)
            cache_dict['config'] = cache_dict['config'].__dict__  # TODO same to all other variables?
            with open(fn, 'wb') as handle:
                pkl.dump(cache_dict, handle)
        except Exception as e:
            raise e

    def from_cache(self):
        """
        basic uncache of all fields
        """
        fn = self.cache_fn
        cached = False
        try:
            import pickle as pkl
            with open(fn, 'rb') as handle:
                uncached_dict = pkl.load(handle)
            # uncache the config variable
            uncached_config_dict = uncached_dict.pop('config')
            uncached_config = TerrainGeneratorConfig(**uncached_config_dict)
            assert self.config == uncached_config
            self.__dict__.update(uncached_dict)
            self.area = self.area_sections.orig_area  # TODO Ed, to keep the same reference
            self.area_sections.update_mock_graph()
            assert self.area is self.area_sections.orig_area
        except Exception as e:
            raise CacheFileNotFoundException('too bad.')

    def test_dijkstra_variants(self, cache: bool = True,
                               noshow: bool = False, save: bool = False):
        if not self.area or not self.area_sections:
            raise Exception("You didn't call generate before running an experiment")
        # # Use vdist for this experiment
        # vdist = self.area_sections.min_vdist
        # self.area_sections.plot_selected_sections(noshow=True,
        #                                           save=(True,
        #                                                 f'experiment{self.cache_fn}.fig_hsections.svg'))  # TODO Ed, remove, because it's a separate thing

        # Restore from cache
        if cache:
            try:
                import pickle as pkl
                with open(f'{self.cache_fn}.dijkstra', 'rb') as handle:
                    return pkl.load(handle)
            except Exception as e:
                print(f'cache "{self.cache_fn}.dijkstra" '
                     f'not found ---> Exception={e}')

        start, target = tuple(self.area.start), tuple(self.area.target)  # TODO Ed, better variant for pts?
        # self.area_sections.update_objective(start, target)  # updates
        if noshow == False:
            self.area_sections.plot_selected_sections(noshow=True,
                                                      save=(True,
                                                      f'{self.cache_fn}.fig_hsections.svg'))  # TODO Ed, remove, because it's a separate thing

            matplotlib.use('TkAgg')
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3,
                                                subplot_kw={"projection": "3d"})
            ax1.set(title='Dijkstra using height distance')
            ax2.set(title='Dijkstra using horizontal distance')
            ax3.set(title='Dijkstra using 3d distance')
            fig.suptitle(f"Road Selection Algorithm\nRoad generated using Config={self.config}")
            # config the figures
            fig.set_figwidth(15)
            fig.set_figheight(10)

        # TODO Ed, turn computation of paths into a different functions, to call it without test_d.._variants

        with timer('dijkstra_by_height()'):
            path1 = self.area_sections.mgraph.dijkstra_by_height(start, target)  # ignore horizontal distance
            if noshow == False:
                self.area_sections.plot_path(path1, fig=fig, ax=ax1)

        with timer('dijkstra_by_length()'):
            path2 = self.area_sections.mgraph.dijkstra_by_length(start, target)  # use both of them
            if noshow == False:
                self.area_sections.plot_path(path2, fig=fig, ax=ax2)

        with timer('dijkstra_by_length_3d()'):
            path3 = self.area_sections.mgraph.dijkstra_by_length_3d(start, target)  # ignore vertical distance
            if noshow == False:
                self.area_sections.plot_path(path3, fig=fig, ax=ax3)

        if noshow == False:
            set_axes_equal(ax1)
            set_axes_equal(ax2)
            set_axes_equal(ax3)
            fig.tight_layout()

            # mng = ax1.get_current_fig_manager()
            # mng.resize(*mng.window.maxsize())
            plt.show()
        if save:
            fig_name = f'{self.cache_fn}.fig_dijkstra.svg'
            fig.savefig(fig_name)
            import pickle as pkl
            with open(f'{fig_name}.3d', 'wb') as handle:
                pkl.dump({
                    "fig": fig,
                    "axes": (ax1, ax2, ax3)
                }, handle)
        result = {
            "height": path1,
            "length": path2,
            "3d": path3
        }
        if cache:
            import pickle as pkl
            with open(f'{self.cache_fn}.dijkstra', 'wb') as handle:
                pkl.dump(result, handle)
        # can you run multiple processes?
        return result

def is_compatible_scaling(grid_size: tuple[int, int], scale_arg: tuple[int, int]) -> bool:
    # TODO Ed, use all() :)
    try:
        answer = grid_size[0] % scale_arg[0] == 0 and grid_size[1] % scale_arg[1] == 0
        return answer
    except:
        return False  # division error


if __name__ == '__main__':

    def grid_based_test(seeds: list[int],
                        GRID_SIZE: list[tuple[int, int]],
                        scaling_argument: list[tuple[int, int]],
                        height_interval: tuple[int, int],
                        height_delta: int):
        experiments = list(filter(
            lambda x: is_compatible_scaling(x[1], x[2]),
            itertools.product(seeds, GRID_SIZE, scaling_argument)))
        cnt = 1
        cnt_experiments = len(experiments)
        for seed, gs, scaling_argument in experiments:
            config = TerrainGeneratorConfig(seed=seed,
                                            GRID_SIZE=gs,
                                            scaling_argument=scaling_argument,
                                            height_delta=height_delta,
                                            height_interval=height_interval)
            e = Experiment(config=config)
            e.generate(cache=True)
            e.reset_objective((5, 5), (gs[0] - 5, gs[1] - 5))
            e.test_dijkstra_variants(noshow=True,
                                     save=True)
            print()
            print(f" >> Experiment {cnt} out of {cnt_experiments} ")
            print()
            cnt += 1


    # grid_based_test(seeds=list(range(5)),
    #                 GRID_SIZE=[(100, 100)],
    #                 scaling_argument=list(zip(range(101),
    #                                           range(101))),
    #                 height_interval=(100, 120),
    #                 height_delta=2)

    # grid_based_test(seeds=[0],
    #                 GRID_SIZE=[(100, 100)],
    #                 scaling_argument=[(2,2)],
    #                 height_interval=(100, 120),
    #                 height_delta=2)

    def interpolation_test():
        config = TerrainGeneratorConfig(seed=2,
                                        GRID_SIZE=(50, 50),
                                        scaling_argument=(2, 2),
                                        height_delta=3,
                                        height_interval=(100, 120))
        e = Experiment(config=config)  # instead of using Experiment, create a class method to also generate data
        e.generate(cache=False)
        # start = (0 + 5,0 + 5)
        # target = (config.GRID_SIZE[0] - 5, config.GRID_SIZE[1] - 5)
        # e.reset_objective(start, target)
        # # REAL CODE:
        # path = e.area_sections.mgraph.dijkstra_by_height(start, target)
        # path3d = [(*coord, e.area_sections.coord_to_height(coord) + .1)
        #      for coord in path]
        # path3d_interpolated = interpolate_3d_path(path3d)
        #
        # matplotlib.use('TkAgg')
        # fig, ax = create_3d_subplots(1,1, figsize=(8, 8))
        #
        # Area._plot_surf_3d(e.area.surf,
        #                    colormap="Blues",
        #                    ax=ax,
        #                    fig=fig,
        #                    alpha=.65)
        # start3d = (*start, e.area_sections.coord_to_height(start))
        # target3d = (*target, e.area_sections.coord_to_height(target))
        # ax.scatter(*zip(*[start3d, target3d]), c=['red'])
        # ax.plot(*zip(*np.array(path3d) + (0,0,2)), c='red', lw=3)
        # ax.plot(*zip(*path3d_interpolated), c='green', lw=3)
        #
        # fig.tight_layout()
        # set_axes_equal(ax)
        #
        # plt.show()

        matplotlib.use('TkAgg')
        fig, ax = create_3d_subplots(1, 1)

        Area._plot_surf_3d(e.area.surf[:20][:, :5],
                           colormap="Blues",
                           ax=ax,
                           fig=fig,
                           alpha=.5)
        path1 = list(zip(range(20),
                         [0] * 20,
                         e.area.surf[:20, 1]))
        path2 = list(zip(range(20),
                         [0] * 20,
                         [e.area.surf[i, j] + 1
                          for i, j in zip(range(20), [0] * 20)]))
        path3 = list(zip(range(20),
                         [0] * 20,
                         [e.area_sections.coord_to_height((i, j)) + 2
                          for i, j in zip(range(20), [0] * 20)]))
        ax.plot(*zip(*path1))
        ax.plot(*zip(*path2))
        ax.plot(*zip(*path3))
        set_axes_equal(ax)
        plt.show()


    # TODO Ed,
    #  This makes the window take up the full screen for me, under Ubuntu 12.04 with the TkAgg backend:
    #  .
    #     mng = plt.get_current_fig_manager()
    #     mng.resize(*mng.window.maxsize())
    #

    # interpolation_test()
    # quit()
    config = TerrainGeneratorConfig(seed=0,
                                    GRID_SIZE=(100, 100),
                                    scaling_argument=(2, 2),
                                    height_interval=(100, 120),
                                    height_delta=2)
    # create experiment
    e = Experiment(config=config)
    e.generate(cache=True)
    e.reset_objective((5, 5), (94, 94))
    # run the simple dijkstra test (Graphing the paths for the random objective
    e.test_dijkstra_variants(noshow=False)

    # TODO Ed, remove old imports of generate_...
    # TODO Ed, 1. do sth with the graph, such as returning the area with a given hdiff, much faster
    #             THEN TEST AND COMMIT
    #          2. method to generate an WeightedGraph, which will implement route planning algorithm
    #             THE TEST THE PLANNING ALGO, AND COMMIT
    #          3. Where to add the routing? Maybe inside another class?

    # TODO Ed, implement interpolation of an area.
