import contextlib
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from auto_path.areas.area_sections import Area, AreaSections
from auto_path.areas.area import interpolate_3d_line, eucl, interpolate_using_bezier_curve

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
#     # matplotlib.use('TkAgg')
#     # plt.interactive(False)
#     # x, y, h = line[:, 0], line[:, 1], line[:, 2]
#     # ax = plt.figure().add_subplot(projection='3d')
#     # ax.plot(x, y, h)
#     # ax.axis('equal')
#     # plt.show()
#     #
#     # dist = [0] + [eucl(a, b)
#     #               for a, b in zip(line[0:, :2],
#     #                               line[1:, :2])]
#     # dist2 = [0] * len(h)
#     # for i in range(1, len(h)):
#     #     dist2[i] = dist2[i - 1] + dist[i]
#     # plt.plot(dist2, h)
#     # plt.axis('equal')
#     # plt.show()
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

from auto_path.areas.graph import generate_hsections_graph


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

if __name__ == '__main__':
    # # compute the full graph
    # slow_method_using_full_graph()

    # compute the HSectionsGraph, which creates areas based on height, computes the contour of each and
    #  stores hsection adjacency data.
    # TODO Ed, include useful methods inside the HSectionsGraph
    with timer('generate area'):
        a = Area.from_perlin_noise(seed=8, GRID_SIZE=(360, 360),
                                   scaling_argument=(3, 3),
                                   min_height=200, max_height=300)
        a.set_objective((35, 70), (333, 209))  # TODO Ed, NEEDED, shouldn't I just move it inside the init method?
    with timer('generate area sections'):
        # TODO Ed, rename this thing, refactor how the from_area function is presented
        #          and add an helper function   surf_to_STH(surf: np.array) -> np.array  ...
        a_sections = AreaSections.from_area(area=a, height_delta=20)
        # a_sections.plot_heatmap()
        vdist = a_sections.min_vdist
        # a_sections.plot_selected_sections(vdist=vdist)

    # generate a full graph:
    with timer('get full graph'):
        wgraph = a_sections.get_full_graph(vdist)

    start, target = tuple(a.start), tuple(a.target)
    with timer('dijkstra_by_height()'):
        path = wgraph.dijkstra_by_height(start, target)  # ignore horizontal distance
        a_sections.plot_path(path)

    with timer('dijkstra_by_length()'):
        path = wgraph.dijkstra_by_length(start, target)  # use both of them
        a_sections.plot_path(path)

    with timer('dijkstra_by_length_3d()'):
        path = wgraph.dijkstra_by_length_3d(start, target)  # ignore vertical distance
        a_sections.plot_path(path)


    # TODO Ed, remove old imports of generate_...
    # TODO Ed, 1. do sth with the graph, such as returning the area with a given hdiff, much faster
    #             THEN TEST AND COMMIT
    #          2. method to generate an WeightedGraph, which will implement route planning algorithm
    #             THE TEST THE PLANNING ALGO, AND COMMIT
    #          3. Where to add the routing? Maybe inside another class?


    # TODO Ed, implement interpolation of an area.