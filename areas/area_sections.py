import functools
import math
from collections import defaultdict
from typing import Hashable, Optional, Iterable, Callable

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

# import area utils
from auto_path.areas.area import Area
from auto_path.areas.graph import Graph, generate_hsections_graph, WeightedGraph, Coord, CoordReal, Coord3dReal, \
    AreaIndex
# import graph utils
from auto_path.areas.graph import generate_hsections_graph
from auto_path.areas.custom_colors import colors

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from auto_path.areas.utils import minmax, are_grid_neighbors, are_corner_neighbors, grid_neighbors_gen, create_subplots, \
    create_3d_subplots


def compute_height_sections(surf: np.array, height_delta: float):
    minn, maxx = surf.min(), surf.max()
    height_delta = int(height_delta)
    buckets = [i for i in range(math.floor(minn), math.ceil(maxx), height_delta)]
    buckets.append(int(maxx))
    find_bucket = lambda x: (x - minn) // height_delta
    find_new_value = lambda x: minn + height_delta * find_bucket(x)
    # print(buckets)
    # print(noise[0, 0],
    #       find_bucket(noise[0, 0]),
    #       find_new_value(noise[0, 0]))
    transform = np.vectorize(find_new_value)
    return transform(surf), buckets


class AreaSections:
    def __init__(self,
                 orig_area: Area,  # TODO Ed, eliminate if possible
                 height_delta: float,
                 generate_full_graphs: bool = False):
        self.orig_area = orig_area
        self.height_delta = height_delta
        # modify the surface data:
        self.surf_h, self.buckets = compute_height_sections(orig_area.surf, height_delta=height_delta)
        # # TODO Ed remove, plot the result:
        # self.plot_heatmap()
        # self.show()

        # TODO may be better to abstractize the list of sets?,
        #  to simply use the ids of any type (int,str etc)
        self.graph = generate_hsections_graph(self.surf_h)
        # vertical dist = number of elevation changes
        self.area_idx_to_vdist: dict[int, int] = {}
        self._map_graphs_to_vdist()
        # # TODO Ed, Deprecated: retaining full graph is not wanted
        # self.full_graphs: dict[int, WeightedGraph] = {}
        # if generate_full_graphs:
        #     self._generate_all_graphs()
        # TODO Ed, new features: setting up the mock graph
        self._mock_graph = AreaSections.MockGraph(self)

    @property
    def mgraph(self):
        return self._mock_graph

    def update_mock_graph(self):
        self._mock_graph = AreaSections.MockGraph(self)

    @property
    def area_ids(self) -> set[int]:
        return {k for k in self.area_idx_to_vdist}

    def available_areas_ids(self, max_vdist: Optional[int] = None) -> set[int]:
        if not max_vdist:
            max_vdist = self.min_vdist
        return {area_idx
                for area_idx, vd in self.area_idx_to_vdist.items()
                if vd <= max_vdist}

    class MockGraph:
        def __init__(self, area_sections: 'AreaSections'):
            self._as = area_sections

        def available_areas_ids(self):  # TODO Ed, necessary?
            return self._as.available_areas_ids()

        def available_areas(self) -> dict[int, set[Coord]]:
            return {idx: self._as.area(idx)
                    for idx in self.available_areas_ids()}

        @property
        def nodes(self) -> set[Coord]:  # TODO Ed, store nodes?
            # union of all available areas
            areas = self.available_areas().values()
            return set(functools.reduce(lambda a1, a2: a1 | a2,
                                        areas))

        def _get_height(self, node: Coord):
            try:
                return self._as.orig_area.surf[node]
            except:
                raise  # or not?

        def _get_weight(self,
                        node1: Coord,
                        node2: Coord) -> tuple[float, float]:
            # TODO Ed, the nodes must be neighbors on the grid, and you compute the horiz and vert distance
            if not are_grid_neighbors(node1, node2, include_corner_neighbors=True):
                return None

            vdist = math.sqrt(2) if are_corner_neighbors(node1, node2) else 1
            hdiff = self._get_height(node1) - self._get_height(node2)
            return hdiff, vdist

        def weight(self,
                   node1: Coord,
                   node2: Coord) -> Optional[tuple[float, float]]:
            return self._get_weight(node1, node2)

        def path_weight(self, path: Iterable[Coord]):  # TODO Ed, useful?
            dh, ln = 0, 0
            for p1, p2 in zip(path,
                              path[1:]):
                hdiff, length = self.weight(p1, p2)
                dh += abs(hdiff)
                ln += length
            return ln, dh

        @classmethod
        def from_area_sections(cls, area_sections: 'AreaSections'):
            return cls(area_sections)

        # TODO Ed, helper methods for graph algos
        def is_edge(self, node1: Coord, node2: Coord):
            """
            Take into account the areas available according to vdist:
                self._as.min_vdist
            """
            if not are_grid_neighbors(node1, node2, include_corner_neighbors=True):
                return False
            vdist_1 = self._as.coord_to_vdist(node1)
            vdist_2 = self._as.coord_to_vdist(node1)
            return vdist_1 <= self._as.min_vdist and vdist_2 <= self._as.min_vdist

        def get_neighbors(self, node: Coord) -> set[Coord]:
            maxij = self._as.orig_area.surf.shape
            neighbors = set(grid_neighbors_gen(*node,
                                               *maxij,
                                               include_corner_neighbors=True))
            neighbors &= self.nodes
            return set(neighbors)

        # add infinite value
        INF = float('inf')  # bacause it measures distances in multiple of meters
        INF_MINUS = float('-inf')  # bacause it measures distances in multiple of meters

        def _dijkstra_with_distance_fn(self,
                                       start: Coord, target: Coord,
                                       dist_fn: Callable = lambda p1, p2: 1) \
                -> tuple[list[Coord], dict]:
            import heapq

            def dijkstra(graph: 'MockGraph', start, end):
                # Create a dictionary to store the distance from start to each node
                distances = {node: float('inf') for node in graph.nodes}
                # Set the distance from start to itself to be 0
                distances[start] = 0
                # Create a dictionary to store the previous node in the shortest path to each node
                previous = defaultdict(lambda: None, {})  # instead of {node: None for node in graph.nodes}
                # Create a priority queue to store nodes with their distances
                pq = [(0, start)]
                # While the priority queue is not empty
                while pq:
                    # Get the node with the smallest distance from the priority queue
                    (dist, node) = heapq.heappop(pq)
                    # If we have already processed this node, continue to the next node
                    if dist > distances[node]:
                        continue
                    # Update the distance to each neighbor of the current node
                    for neighbor in graph.get_neighbors(node):  # TODO Ed, implement WeightedGraph.node(idx)
                        weight = dist_fn(node, neighbor)
                        distance = dist + weight
                        # If the distance to the neighbor through this node is smaller than the current distance to the neighbor
                        if distance < distances[neighbor]:
                            # Update the distance to the neighbor
                            distances[neighbor] = distance
                            # Set the previous node in the shortest path to be the current node
                            previous[neighbor] = node
                            # Add the neighbor and its distance to the priority queue
                            heapq.heappush(pq, (distance, neighbor))
                # If there is no path from start to end, return None
                if previous[end] is None:
                    return None, distances
                # Construct the shortest path from start to end by following the previous nodes backwards from end to start
                path = [end]
                while previous[path[0]] is not None:
                    path.insert(0, previous[path[0]])
                return path, distances

            # graph = {
            #     'A': {'B': 2, 'C': 4},
            #     'B': {'C': 1, 'D': 2},
            #     'C': {'D': 1},
            #     'D': {}
            # }
            # path, distances = dijkstra(graph, 'A', 'D')
            # print(distances)  # {'A': 0, 'B': 2, 'C': 3, 'D': 4}
            # print(path)  # ['A', 'B', 'D']

            path, distances = dijkstra(self, start=start, end=target)
            return path, distances

        def _is_inf(self, val: float):
            return math.isinf(val)

        def dijkstra_by_height(self, start: Coord, target: Coord):
            """
            """

            def distance(node1: Coord, node2: Coord):
                from math import sqrt
                hdist, vdist = self.weight(node1, node2)
                dist = abs(hdist)
                return dist if self.is_edge(node1, node2) \
                    else self.INF

            path, distances = self._dijkstra_with_distance_fn(start=start, target=target,
                                                              dist_fn=distance)
            if not path:
                raise Exception('There was no path between the two objectives')
            return path

        def dijkstra_by_length(self, start: Coord, target: Coord):
            """
            """

            def distance(p1: Coord, p2: Coord):
                from math import sqrt
                a, b = np.array([p1, p2])
                dist = sqrt(np.sum((a - b) ** 2))
                return dist if self.is_edge(p1, p2) \
                    else self.INF

            path, distances = self._dijkstra_with_distance_fn(start=start, target=target,
                                                              dist_fn=distance)
            if not path:
                raise Exception('There was no path between the two objectives')
            return path

        def dijkstra_by_length_3d(self, start: Coord, target: Coord):
            """
            """

            def distance(node1: Coord, node2: Coord):
                from math import sqrt
                hdist, vdist = self.weight(node1, node2)
                dist_3d = sqrt(hdist ** 2 + vdist ** 2)
                return dist_3d if self.is_edge(node1, node2) \
                    else self.INF

            path, distances = self._dijkstra_with_distance_fn(start=start, target=target,
                                                              dist_fn=distance)
            if not path:
                raise Exception('There was no path between the two objectives')
            return path

    def update_objective(self,
                         start: Coord,
                         target: Coord):
        self.orig_area.set_objective(start, target)
        self._map_graphs_to_vdist()

    def shortest_path_areas_ids(self):  # TODO Ed, necessary?
        path = self.graph.dijkstra_for_objective(
            self.coord_to_area_idx(self.start),
            self.coord_to_area_idx(self.target)
        )  # TODO Ed, create dijkstra for areas
        return set(path)

    def coord_to_vdist(self, coord: Coord):
        a_idx = self.coord_to_area_idx(coord)
        return self.area_idx_to_vdist[a_idx]

    def _map_graphs_to_vdist(self):
        # TODO Ed, instead of generating them all,
        #  just note down for each area the number of elevation changes to reach it
        # use BFS starting in bots start and target
        self.area_idx_to_vdist: dict[int, int] = {}

        a1_idx = self.coord_to_area_idx(self.start)
        a2_idx = self.coord_to_area_idx(self.start)
        # initialise q and vdist for the start point
        q = [a1_idx]
        self.area_idx_to_vdist[a1_idx] = 0

        while q:
            a_idx = q.pop(-1)
            vdist = self.area_idx_to_vdist.get(a_idx)
            # get neightboring areas
            for a2_idx in self.graph.get_neighbors(area_idx=a_idx):
                if self.area_idx_to_vdist.get(a2_idx) is None:
                    q.append(a2_idx)
                    self.area_idx_to_vdist[a2_idx] = vdist + 1
        # print(self.area_idx_to_vdist)

    def _generate_all_graphs(self):
        # # TODO Ed, instead of generating them all,
        # #  just note down for each area the number of elevation changes to reach it
        # # use BFS starting in bots start and target
        # q = [self.start, self.target]
        pass

    # TODO Ed, property for fetching one graph based on vdist
    def get_full_graph(self, vdist: int):  # TODO Ed, do not use for the experiments
        available_area = functools.reduce(
            lambda a1, a2: a1 | a2,
            [
                self.area(area_idx)
                for area_idx in self.graph.nodes
                if self.area_idx_to_vdist.get(area_idx) <= vdist
            ],
            set())
        surf = self.orig_area.surf
        # TODO Ed, keep the graph in memory
        self.full_graphs[vdist] = WeightedGraph.from_available_surface(available_area, surf)
        return self.full_graphs[vdist]

    def area(self, area_idx) -> set[Coord]:
        return self.graph.hsection(area_idx).pts

    @property
    def areas(self):
        return self.graph.hsections

    def coord_to_area(self, coord: Coord) -> set[Coord]:
        area_idx = self.graph.coord_to_area_idx(coord)
        area = self.graph.hsection(area_idx).pts
        return area

    def coord_to_area_idx(self, coord: tuple[int, int]) -> AreaIndex:
        return self.graph.coord_to_area_idx(coord)

    def coord_to_height(self, coord: Coord) -> float:
        return self.orig_area.surf[coord]

    # TODO Ed, added interpolation of height:
    def interpolate_height(self, coord: CoordReal):
        return self.orig_area.interpolate_height(np.array(coord))

    # TODO Ed, added interpolation of height:
    def interpolate_path_height(self, path: list[CoordReal]) -> list[Coord3dReal]:
        return [(*coord, self.orig_area.interpolate_height(np.array(coord)))
                for coord in path]

    def coord_to_height_bucket(self, coord: Coord) -> float:
        return self.surf_h[coord]

    def area_to_height(self, area_idx: int):
        area = self.graph.hsection(area_idx).pts
        pt_in_area = area
        return self.surf_h[pt_in_area]

    @classmethod
    def from_area(cls, area: Area, height_delta: float):
        return AreaSections(orig_area=area, height_delta=height_delta)

    @property
    def start(self):
        return self.orig_area.start

    @property
    def target(self):
        return self.orig_area.target

    def show(self):
        # TODO Ed, may do a superclass named plotable? :)
        #  may also add an ax/fig configuration to enable printing with a given axis such as 'equal'
        plt.show()

    def _plot_surf(self, surf: np.array,
                   start: np.array = None,
                   target: np.array = None,
                   fig: Figure = None, ax: Axes = None,
                   flip: bool = True,
                   plot_objective: bool = True):
        if not fig and not ax:
            pass  # TODO Ed, create them
        surf = surf.T if flip else surf
        # plot image
        # im = ax.imshow(surf, cmap='Blues', interpolation='antialiased')  # TODO Ed, IT DOES INTERPOLATE
        im = ax.imshow(surf, cmap='Blues', interpolation='nearest')  # TODO Ed, IT DOES INTERPOLATE
        # create custom colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.15)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        # plot objective TODO Ed, pass more args
        if plot_objective:
            self._plot_objective_3d(fig=fig, ax=ax)

    def _plot_objective(self,
                        start: np.array = None,
                        target: np.array = None,
                        fig: Figure = None,
                        ax: Axes = None):
        if not fig and not ax:
            pass  # TODO Ed, create them
        if start is None:
            start = self.start
        if target is None:
            target = self.target
        if start is not None and target is not None:
            scatter_data = list(zip(*[start, target]))
            ax.scatter(x=scatter_data[0], y=scatter_data[1], c=['red'])
            ax.text(start[0] - 5,
                    start[1] - 2, s=' start', fontdict={'family': 'sans-serif',
                                                        'size': 16,
                                                        'weight': 'bold',
                                                        'color': 'red'})
            ax.text(target[0] - 7.5,
                    target[1] - 2, s='target', fontdict={'family': 'sans-serif',
                                                         'size': 16,
                                                         'weight': 'bold',
                                                         'color': 'red'})

    def _plot_objective_3d(self,
                           start3d: np.array = None,
                           target3d: np.array = None,
                           fig: Figure = None,
                           ax: Axes = None):
        if not fig and not ax:
            pass  # TODO Ed, create them
        if start3d is None:
            start3d = self.start  # TODO 3D
        if target3d is None:
            target3d = self.target  # TODO 3D
        # add height to start,target
        start3d[0], start3d[1] = start3d[1], start3d[0]  # TODO Ed, 2D
        target3d[0], target3d[1] = target3d[1], target3d[0]  # TODO Ed, 2D
        if start3d is not None and target3d is not None:
            scatter_data = list(zip(*[start3d, target3d]))
            ax.scatter(scatter_data[0],
                       scatter_data[1],
                       scatter_data[2],
                       c=['red'])
            ax.text(*(start3d + (-5, 0, 5)), s=' start',
                    fontdict={'family': 'sans-serif',  # TODO Ed, set position, orientation etc
                              'size': 14,
                              'weight': 'bold',
                              'color': 'red'})
            ax.text(*(target3d + (-5, 0, 5)), s='target', fontdict={'family': 'sans-serif',
                                                                    'size': 14,
                                                                    'weight': 'bold',
                                                                    'color': 'red'})

    def _plot_objective_2d(self,
                           start: np.array = None,
                           target: np.array = None,
                           fig: Figure = None,
                           ax: Axes = None):
        if not fig and not ax:
            fig, ax = create_subplots(1, 1)
        if start is None:
            start = np.array(self.start)
        if target is None:
            target = np.array(self.target)
        # add height to start,target
        start[0], start[1] = start[1], start[0]
        target[0], target[1] = target[1], target[0]
        if start is not None and target is not None:  # NOT NEEDED?
            scatter_data = list(zip(*[start, target]))
            ax.scatter(scatter_data[0],
                       scatter_data[1],
                       c=['black'])
            ax.text(*(start + (-4, -1)), s=' start',
                    fontdict={'family': 'sans-serif',  # TODO Ed, set position, orientation etc
                              'size': 18,
                              'weight': 'bold',
                              'color': 'black'})
            ax.text(*(target + (-8, 3)), s='target', fontdict={'family': 'sans-serif',
                                                               'size': 18,
                                                               'weight': 'bold',
                                                               'color': 'black'})

    def plot_heatmap(self, fig: Figure = None,
                     ax: Axes = None,
                     plot_objective: bool = True):  # TODO Ed, enable mpl backend update using a method which returns self
        if self.surf_h is None:
            raise NotImplementedError('You should first run compute_height_sections')
        self._plot_surf(self.surf_h, fig=fig, ax=ax,
                        plot_objective=plot_objective)

    def plot_areas(self):  # TODO Ed, is this still useful?
        # self._plot_surf(self.surf_h, self.start, self.target)
        for i, area in enumerate(self.areas):
            area = [pt[::-1] for pt in area]
            plt.scatter(*zip(*area), edgecolors='none', c=[colors[i % len(colors)]],
                        marker=',', lw=0, s=1)  # scatter only one pixel
        # plt.axis('equal')
        plt.scatter(*zip(*[self.start, self.target]), c=['red'])
        plt.show()

    @property
    def min_vdist(self) -> int:
        target_area_idx = self.coord_to_area_idx(self.target)
        return self.area_idx_to_vdist.get(target_area_idx)

    def plot_selected_sections(self,
                               ax: Axes = None,
                               fig: Figure = None,
                               noshow: bool = False,
                               plot_objective: bool = True,
                               save: tuple[bool, str] = (False, '')):
        if self.start is None:
            raise ValueError("self.start was unset")
        if self.target is None:
            raise ValueError("self.end was unset")
        # TODO Ed, plot heatmap. DOES THIS flip?
        if not fig or not ax:
            fig, ax = plt.subplots(1, 1)
        self.plot_heatmap(fig=fig, ax=ax,
                          plot_objective=plot_objective)

        for idx in self.shortest_path_areas_ids():
            area = self.area(idx)
            # TODO Ed, color based on difference to start (start is at middle, and you go up and down
            ax.scatter(*zip(*area), facecolor='none', c=['gold'],
                       marker='x', lw=1, s=3)  # scatter only one pixel
            # TODO for better images, print using polygons, you just need the contours for each area.
        # scatter the objective points and their name
        if plot_objective:
            self._plot_objective(fig=fig, ax=ax)
        if not noshow:
            plt.show()
        if save[0]:
            plt.savefig(save[1])

    # TODO Ed, add function for plotting the 3D path along the terrain
    def plot_path_3d(self,
                     path: list[Coord],
                     ax: Axes3D,
                     fig: Figure):
        plt.interactive(False)
        # get data:
        x, y = zip(*path)
        h = [self.coord_to_height(coord) + 0  # TODO Ed, will remove after finding out how to plot on top of the surface
             for coord in path]
        # create figure and 3d axes
        if not fig or not ax:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # plot terrain
        # TODO Ed, create a class and helper functions for easier plotting, grid snippets and so on :)
        dim1, dim2 = self.orig_area.surf.shape
        minx, maxx = minmax([pt[0] for pt in path])
        miny, maxy = minmax([pt[1] for pt in path])
        border = min(20,
                     minx, miny,
                     dim1 - minx, dim2 - miny)
        minx -= border
        miny -= border
        maxx += border
        maxy += border
        partial_surf = self.orig_area.surf[minx:maxx,
                       miny:maxy]
        Area._plot_surf_3d(partial_surf,
                           colormap="Blues",
                           ax=ax,
                           fig=fig,
                           alpha=.5)
        # plot path  # TODO Ed, also transform Coord to start with 0,0. Also include some margin
        # TODO Ed, this could be a functiona
        x = [xcoord - minx + border // 2
             for xcoord in x]
        y = [ycoord - miny + border // 2
             for ycoord in y]
        ax.plot(y, x, h[::-1], c='red', lw=2)  # TODO Ed, why?
        # add start and target points
        start = np.array(self.start) + (- minx + border // 2, - miny + border // 2)  # TODO Ed, why?
        target = np.array(self.target) + (- minx + border // 2, - miny + border // 2)  # TODO Ed, why?
        start3d = np.array([*start, self.coord_to_height(tuple(start))])
        target3d = np.array([*target, self.coord_to_height(tuple(target))])
        self._plot_objective_3d(fig=fig,
                                ax=ax,
                                start3d=start3d,
                                target3d=target3d)
        # # reset backend to pycharm
        # matplotlib.use('module://backend_interagg')

    def plot_path_3d_real(self,
                          path: list[Coord],
                          ax: Axes3D = None,
                          fig: Figure = None):
        plt.interactive(False)
        # get data:
        x, y, h = zip(*self.interpolate_path_height(path))
        # create figure and 3d axes
        if not fig or not ax:
            fig, ax = create_3d_subplots(1, 1)
        # plot terrain
        # TODO Ed, create a class and helper functions for easier plotting, grid snippets and so on :)
        dim1, dim2 = self.orig_area.surf.shape
        minx, maxx = minmax([pt[0] for pt in path])
        miny, maxy = minmax([pt[1] for pt in path])
        border = min(20,
                     minx, miny,
                     dim1 - minx, dim2 - miny)
        minx -= border
        miny -= border
        maxx += border
        maxy += border
        partial_surf = self.orig_area.surf[int(minx):int(maxx),
                                           int(miny):int(maxy)]
        Area._plot_surf_3d(partial_surf,
                           colormap="Blues",
                           ax=ax,
                           fig=fig,
                           alpha=.5)
        # plot path  # TODO Ed, also transform Coord to start with 0,0. Also include some margin
        # TODO Ed, this could be a functiona
        x = [xcoord - minx + border // 2
             for xcoord in x]
        y = [ycoord - miny + border // 2
             for ycoord in y]
        ax.plot(y, x, h[::-1], c='red', lw=4)  # TODO Ed, why?
        # add start and target points
        start = np.array(self.start) + (- minx + border // 2, - miny + border // 2)  # TODO Ed, why?
        target = np.array(self.target) + (- minx + border // 2, - miny + border // 2)  # TODO Ed, why?
        start3d = np.array([*start, self.interpolate_height(np.array(start))])
        target3d = np.array([*target, self.interpolate_height(np.array(target))])
        self._plot_objective_3d(fig=fig,
                                ax=ax,
                                start3d=start3d,
                                target3d=target3d)
        # # reset backend to pycharm
        # matplotlib.use('module://backend_interagg')

    def plot_path_2d(self,
                     path: list[CoordReal],
                     ax: Axes,
                     fig: Figure,
                     border: int = 20):
        # get data:
        x, y = zip(*path)
        # create figure and 3d axes
        if not fig or not ax:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # plot terrain
        # TODO Ed, create a class and helper functions for easier plotting, grid snippets and so on :)
        dim1, dim2 = self.orig_area.surf.shape
        minx, maxx = minmax([pt[0] for pt in path])
        miny, maxy = minmax([pt[1] for pt in path])
        border = min(20,
                     minx, miny,
                     dim1 - minx, dim2 - miny)
        minx -= border
        miny -= border
        maxx += border
        maxy += border
        self.plot_selected_sections(noshow=True,
                                    ax=ax, fig=fig,
                                    plot_objective=False)
        # plot path  # TODO Ed, also transform Coord to start with 0,0. Also include some margin
        # TODO Ed, this could be a functiona
        ax.plot(x, y, c='red', lw=2)  # TODO Ed, why?
        # add start and target points
        start = np.array(self.start)  # + (- minx + border // 2, - miny + border // 2)  # TODO Ed, why?
        target = np.array(self.target)  # + (- minx + border // 2, - miny + border // 2)  # TODO Ed, why?
        self._plot_objective_2d(fig=fig,
                                ax=ax,
                                start=start,
                                target=target)

    def plot_grouped(self, max_hdiff: float):  # TODO Ed, older function
        if self.start is None:
            raise ValueError("self.start was unset")
        if self.target is None:
            raise ValueError("self.end was unset")
        # get height of each area, to compute the height interval
        start = tuple(self.start)
        target = tuple(self.target)
        h1, h2 = self.coord_to_height(start), \
            self.coord_to_height(target)
        if h1 > h2:
            h1, h2 = h2, h1
        hmin, hmax = h1 - max_hdiff, h2 + max_hdiff

        # merge area that are adjacent to the area for start/target
        # and have a minimal/maximal height
        lt, rt = 0, 1
        selected = [self.coord_to_area_idx(start),
                    self.coord_to_area_idx(target)]  # TODO Ed, could be faster

        while lt <= rt:
            curr_idx = selected[lt]
            neighbors = [area_idx for area_idx in self.graph.get_neighbors(curr_idx)
                         if area_idx not in selected and \
                         hmin <= self.area_to_height(area_idx) <= hmax]
            selected.extend(neighbors)
            # update pointers
            lt += 1
            rt += len(neighbors)
        selected_area_ids = set(selected)
        print(selected_area_ids)

        self.plot_heatmap()
        colors_light_to_dark = [
            'white',
            'ivory',
            'lightyellow',
            'lemonchiffon',
            'beige',
            'blanchedalmond',
            'moccasin',
            'navajowhite',
            'burlywood',
            'goldenrod',
            'darkgoldenrod',
            'olive',
            'darkolivegreen',
            'darkgreen',
            'darkslategray',
            'black'
        ]
        MIDDLE = 5  # 'moccasin'
        start_height = self.coord_to_height(tuple(self.start))  # TODO Ed, call tuple?
        for i, idx in enumerate(selected_area_ids):
            area = self.areas[idx]
            a_coord = next(iter(area))
            hdiff = abs(math.floor((self.coord_to_height(
                a_coord) - start_height) / self.height_delta))  # works in this case, but not always,
            # because TODO Ed, target and start may have different height
            plt.scatter(*zip(*area), edgecolors='none', c=[colors_light_to_dark[hdiff]],
                        # TODO Ed, side: plot them using c=
                        marker=',', lw=0, s=1)  # scatter only one pixel
            # TODO for better images, print using polygons, you just need the contours for each area.
        # plt.axis('equal')
        plt.scatter(*zip(*[self.start, self.target]), c=['red'])
        plt.show()
