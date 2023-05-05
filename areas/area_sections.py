import functools
import math
from collections import defaultdict

# import area utils
from auto_path.areas.area import Area
from auto_path.areas.graph import Graph, generate_hsections_graph, WeightedGraph, Coord
# import graph utils
from auto_path.areas.graph import generate_hsections_graph
from auto_path.areas.custom_colors import colors

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from auto_path.areas.utils import minmax


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
        self.full_graphs: dict[int, WeightedGraph] = defaultdict(lambda: None, {})
        self.area_idx_to_vdist: dict[int, int] = defaultdict(lambda: None, {})
        if generate_full_graphs:
            self._generate_all_graphs()
        self._map_graphs_to_vdist()

    def _map_graphs_to_vdist(self):
        # TODO Ed, instead of generating them all,
        #  just note down for each area the number of elevation changes to reach it
        # use BFS starting in bots start and target
        self.area_idx_to_vdist: dict[int, int] = defaultdict(lambda: None, {})

        a1_idx = self.coord_to_area_idx(self.start)
        a2_idx = self.coord_to_area_idx(self.start)
        # initialise q and vdist for the start point
        q = [a1_idx]
        self.area_idx_to_vdist[a1_idx] = 0

        while q:
            a_idx = q.pop(-1)
            vdist = self.area_idx_to_vdist[a_idx]
            # get neightboring areas
            for a2_idx in self.graph.get_neighbors(area_idx=a_idx):
                if self.area_idx_to_vdist[a2_idx] is None:
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
    def get_full_graph(self, vdist: int):
        available_area = functools.reduce(
            lambda a1, a2: a1 | a2,
            [
                self.area(area_idx)
                for area_idx in self.graph.nodes
                if self.area_idx_to_vdist[area_idx] <= vdist
            ],
            set())
        surf = self.orig_area.surf
        # TODO Ed, keep the graph in memory
        self.full_graphs[vdist] = WeightedGraph.from_available_surface(available_area, surf)
        return self.full_graphs[vdist]

    def area(self, area_idx) -> set[Coord]:
        return self.graph.hsection(area_idx).pts

    def coord_to_area(self, coord: Coord) -> set[Coord]:
        area_idx = self.graph.coord_to_area_idx(coord)
        area = self.graph.hsection(area_idx).pts
        return area

    def coord_to_area_idx(self, coord: tuple[int, int]) -> int:
        return self.graph.coord_to_area_idx(coord)

    def coord_to_height(self, coord: Coord) -> float:
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
                   start: np.array, target: np.array,
                   flip: bool = True):
        surf = surf.T if flip else surf
        plt.imshow(surf, cmap='Blues', interpolation='lanczos')
        plt.colorbar()
        plt.scatter(*zip(*[start, target]), c=['red'])

    def plot_heatmap(self):  # TODO Ed, enable mpl backend update using a method which returns self
        if self.surf_h is None:
            raise NotImplementedError('You should first run compute_height_sections')
        self._plot_surf(self.surf_h, self.start, self.target)

    def plot_areas(self):  # TODO Ed, is this still useful?
        self._plot_surf(self.surf_h, self.start, self.target)
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
        return self.area_idx_to_vdist[target_area_idx]

    def plot_selected_sections(self, vdist: int):
        if self.start is None:
            raise ValueError("self.start was unset")
        if self.target is None:
            raise ValueError("self.end was unset")

        selected_area_ids = {area_idx
                             for area_idx, vd in self.area_idx_to_vdist.items()
                             if vd <= vdist}
        print(selected_area_ids)
        self.plot_heatmap()
        # colors_light_to_dark = [
        #     'white',
        #     'ivory',
        #     'lightyellow',
        #     'lemonchiffon',
        #     'beige',
        #     'blanchedalmond',
        #     'moccasin',
        #     'navajowhite',
        #     'burlywood',
        #     'goldenrod',
        #     'darkgoldenrod',
        #     'olive',
        #     'darkolivegreen',
        #     'darkgreen',
        #     'darkslategray',
        #     'black'
        # ]
        for idx in selected_area_ids:
            area = self.area(idx)
            # TODO Ed, color based on difference to start (start is at middle, and you go up and down
            plt.scatter(*zip(*area), edgecolors='none', c=['moccasin'],
                        marker='.', lw=0, s=2)  # scatter only one pixel
            # TODO for better images, print using polygons, you just need the contours for each area.
        # plt.axis('equal')
        plt.scatter(*zip(*[self.start, self.target]), c=['red'], s=27)
        plt.show()

    # TODO Ed, add function for plotting the 3D path along the terrain
    def plot_path(self, path: list[Coord]):
        matplotlib.use('TkAgg')
        plt.interactive(False)
        # get data:
        x, y = zip(*path)
        h = [self.orig_area.surf[x, y] + 10  # TODO Ed, will remove after finding out how to plot on top of the surface
             for x, y in path]
        # create figure and 3d axes
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # plot terrain
        # TODO Ed, create a class and helper functions for easier plotting, grid snippets and so on :)
        dim1, dim2 = self.orig_area.surf.shape
        minx, maxx = minmax([pt[0] for pt in path])
        miny, maxy = minmax([pt[1] for pt in path])
        border = min(20,
                     minx, miny,
                     dim1 - minx, dim2 - miny)
        partial_surf = self.orig_area.surf[minx - border:maxx + border + 1,
                                           miny - border:maxy + border + 1]
        Area._plot_surf_3d(partial_surf,
                           colormap="Blues",
                           ax=ax,
                           fig=fig,
                           alpha=.6)
        # plot path  # TODO Ed, also transform Coord to start with 0,0. Also include some margin
        x = [xcoord - minx + border // 2
             for xcoord in x]
        y = [ycoord - miny + border // 2
             for ycoord in y]
        ax.plot(y, x, h, c='red', lw=2)  # TODO Ed, bug, why flip?
        ax.axis('equal')
        plt.show()
        # reset backend to pycharm
        matplotlib.use('module://backend_interagg')

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
