import itertools
import math
from collections import defaultdict
from types import NoneType

from typing import Union, Optional, Iterable, Hashable, Callable

import numpy as np

from areas.utils import grid_neighbors_gen

# Types:


Coord = tuple[int, int]
CoordReal = tuple[float, float]
Coord3D = tuple[int, int, int]
Coord3dReal = tuple[float, float, float]
HSectionIndexPair = tuple[int, int]
AreaIndex = Union[int, NoneType]
CoordToAreaIndex = dict[Coord, AreaIndex]

SharedContour = dict[HSectionIndexPair, set[Coord]]


# class for retaining a Graph
class Graph:
    def __init__(self, num_nodes: int = None):
        self.num_nodes = num_nodes
        self.edges: dict[int, set[int]] = {}

    def add_edge(self, node1: int, node2: int):
        if not self.num_nodes:
            raise Exception("num_nodes was not set")
        if not 0 <= node1 < self.num_nodes:
            raise Exception(f"node1={node1} should be between 0 and {self.num_nodes - 1}")
        if not 0 <= node2 < self.num_nodes:
            raise Exception(f"node2={node2} should be between 0 and {self.num_nodes - 1}")
        self._connect_nodes(node1, node2)

    def _connect_nodes(self, node1, node2):
        if node1 == node2:
            return
        self.edges[node1].add(node2)
        self.edges[node2].add(node1)

    def get_neighbors(self, node: int) -> list[int]:
        return list(self.edges.get(node))


class WeightedGraph(Graph):  # TODO Ed, IMPORTANT: is this graph usable only for nodes of type Coord?
    def __init__(self,
                 nodes: Iterable[Hashable],
                 *args, **kwargs):
        self.nodes = set(nodes)
        """
        self._weight[(coord1, coord2)] is (hdiff, vdiff)
        """
        self._weight = {}
        # use other type of edges:
        self.edges: dict[Coord, set[Coord]] = {}
        super().__init__(*args, **kwargs)

    def _connect_nodes_weighted(self,
                                node1: Hashable,
                                node2: Hashable,
                                weight: tuple[float, float]):
        super()._connect_nodes(node1, node2)
        self.update_weight(node1, node2, weight)

    def add_edge(self,
                 node1: Hashable,
                 node2: Hashable,
                 weight: tuple[float, float] = 0):  # TODO Ed, should I imlement with default value as infinity?
        if node1 not in self.nodes or node2 not in self.nodes:
            raise Exception(f"{node1} or {node2} is not part of {self.nodes}")
        self._connect_nodes_weighted(node1, node2, weight)

    def weight(self,
               node1: Hashable,
               node2: Hashable) -> Optional[tuple[float, float]]:
        return self._weight.get((node1, node2), None)

    def update_weight(self,
                      node1: Hashable,
                      node2: Hashable,
                      weight: tuple[float, float]):
        self._weight[(node1, node2)] = weight
        self._weight[(node2, node1)] = (weight[0], -weight[1])

    @classmethod
    def from_surface(cls,
                     surf: np.array):

        # # try open the cache:
        #
        # with open('cached_graph.json', 'r') as f:
        #
        #     import json
        #     graph = json.loads(f.read())
        # with open('cached_graph.json', 'w') as f:
        #     from copy import deepcopy
        #     graph_data = deepcopy(wgraph.__dict__)
        #     # def as_serializable(d: dict):
        #     #     # tuple keys are converted to str
        #     #
        #     #     # set values are converted to lists
        #     #
        #     # TODO Ed, best variant: serializing library
        #     #          fastest: interpret ast.literal_eval, repr, and then eval for deserializing
        #     import json
        #     f.write(json.dumps(wg.__dict__))

        # Basic uncache using pickle
        cached = False
        try:
            import pickle as pkl
            with open('full_graph.pkl', 'rb') as handle:
                wgraph_dict = pkl.load(handle)
            wgraph = WeightedGraph(nodes=[])
            wgraph.__dict__.update(wgraph_dict)
            # TODO show messages?
            cached = True
        except Exception as e:
            dim1, dim2 = surf.shape
            # the ids are 0..len(surf)-1 and 0..len(surf[0])-1
            nodes = set([tuple(item)
                         for item in itertools.product(range(dim1),
                                                       range(dim2))])
            wgraph = WeightedGraph(nodes=nodes, num_nodes=len(nodes))

            for i, j in itertools.product(range(dim1),
                                          range(dim2)):
                for i2, j2 in grid_neighbors_gen(i, j, dim1, dim2,
                                                 include_corner_neighbors=True):
                    try:
                        hdiff = surf[i2, j2] - surf[i, j]
                        weight = (hdiff, 1)
                        wgraph.add_edge((i, j), (i2, j2), weight)
                    except:
                        pass

        # Basic cache using pickle
        if not cached:
            # Cache it for the next run
            try:
                import pickle as pkl
                with open('full_graph.pkl', 'wb') as f:
                    pkl.dump(wgraph.__dict__, f, protocol=pkl.HIGHEST_PROTOCOL)
            except:
                raise

        return wgraph

    # TODO Ed, algorithm for caching the graph. Save both surf and graph, to be able to easily recover it
    #          or base it aroudn the unique seed? :) save seed inside a private field
    #          THE SECOND ONE seems to be the best

    # TODO Continue next two implementations,
    #      Implement graph path planning
    def shortest_path(self, a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
        # function used to test the basic implementation
        path = [a]
        while a != b:
            dx, dy = b[0] - a[0], b[1] - a[1]
            if dx:
                dx //= abs(dx)
            if dy:
                dy //= abs(dy)
            a = (a[0] + dx, a[1] + dy)
            path.append(a)

        return path

    def path_weight(self, path: Iterable[tuple[int, int]]):
        ln, dh = 0, 0
        for p1, p2 in zip(path,
                          path[1:]):
            hdiff, length = self.weight(p1, p2)
            ln += length
            dh += abs(hdiff)
        return ln, dh

    @classmethod
    def from_available_surface(cls, available_area: set[Coord], surf: np.array):
        # TODO pass use only the nodes from available_area, and
        #  the surf to compute the distances
        # Basic uncache using pickle
        cached = False
        try:
            raise Exception()
            import pickle as pkl
            with open('full_graph_vdist.pkl', 'rb') as handle:
                wgraph_dict = pkl.load(handle)
            wgraph = WeightedGraph(nodes=[])
            wgraph.__dict__.update(wgraph_dict)
            # TODO show messages?
            cached = True
        except Exception as e:
            dim1, dim2 = surf.shape
            # the ids are 0..len(surf)-1 and 0..len(surf[0])-1
            wgraph = WeightedGraph(nodes=available_area,
                                   num_nodes=len(available_area))
            for i, j in available_area:
                available_neighbors = set(
                    grid_neighbors_gen(i, j, dim1, dim2,
                                       include_corner_neighbors=True)
                ) & available_area
                for i2, j2 in available_neighbors:
                    try:
                        hdiff = surf[i2, j2] - surf[i, j]
                        vdiff = 1 if (i == i2 or j == j2) else math.sqrt(2)
                        weight = (hdiff, vdiff)
                        wgraph.add_edge((i, j), (i2, j2), weight)
                    except:
                        pass  # Exception happens when i2,j2 is not part of surf

        # Basic cache using pickle
        if not cached:
            # Cache it for the next run
            try:
                import pickle as pkl
                with open('full_graph_vdist.pkl', 'wb') as f:
                    pkl.dump(wgraph.__dict__, f, protocol=pkl.HIGHEST_PROTOCOL)
            except:
                raise

        return wgraph

    # TODO Ed, helper methods for graph algos
    def is_edge(self, node1: Coord, node2: Coord):
        return node1 in self.edges[node2]

    def get_neighbors(self, node: Coord):
        if not self.edges.get(node):
            self.edges[node] = set()
        return self.edges[node]

    # add infinite value
    INF = float('inf')  # bacause it measures distances in multiple of meters
    INF_MINUS = float('-inf')  # bacause it measures distances in multiple of meters

    def _dijkstra_with_distance_fn(self,
                                   start: Coord, target: Coord,
                                   dist_fn: Callable = lambda p1, p2: 1) \
            -> tuple[list[Coord], dict]:
        import heapq

        def dijkstra(graph, start, end):
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


# class ProxyWGraph(WeightedGraph):
#     def __init__(self, graph: HSectionsGraph,
#                  area_idx_to_vdist: dict[int,int],
#                  ):
#         self._area_sections = area_sections
#         super().__init__(*kargs, *kwargs)

# util functions for graphs
class HSection:
    def __init__(self,
                 pts: set[tuple[int, int]],
                 contour: set[tuple[int, int]]):
        self.pts = pts
        self.contour = contour


class HSectionsGraph:
    """
    stores the nodes as HSection objects.
    Also store coord_to_section_idx   mapping
    """

    def __init__(self,
                 nodes: Iterable[Hashable],
                 hsections: list[HSection],
                 shared_contour: SharedContour,
                 coord_to_area_idx: CoordToAreaIndex):
        self.nodes = set(nodes)
        self._hsections = hsections
        self._num_nodes = len(self.nodes)
        self._shared_contour = shared_contour
        self._edges: dict[AreaIndex, set[AreaIndex]] = {}
        self._coord_to_area_idx = coord_to_area_idx

    def coord_to_area_idx(self, coord: Coord) -> AreaIndex:
        return self._coord_to_area_idx.get(coord)

    def get_neighbors(self, area_idx: int) -> set[AreaIndex]:
        # TODO Ed, this may be upgraded, because you have to iterate all edges...
        return self._edges[area_idx].copy()

    def add_edge(self,
                 node1: AreaIndex,
                 node2: AreaIndex):
        errors = []
        if node1 not in self.nodes:
            errors.append(f"{node1} is not part of self.nodes={self.nodes}")
        if node2 not in self.nodes:
            errors.append(f"{node2} is not part of self.nodes={self.nodes}")
        if errors:
            raise Exception('Error while adding HSectionsGraph edge: ' + '\n  '.join(errors))
        if not self._edges.get(node1):
            self._edges[node1] = set()
        self._edges[node1].add(node2)
        if not self._edges.get(node2):
            self._edges[node2] = set()
        self._edges[node2].add(node1)

    def hsection(self, idx: int) -> HSection:
        return self._hsections[idx]

    @property
    def hsections(self) -> list[HSection]:
        return self._hsections

    # TODO Ed, other useful functions, such as:
    def dijkstra_for_objective(self, start: AreaIndex, target: AreaIndex) -> list[AreaIndex]:
        import heapq
        def dist_fn(node1: AreaIndex, node2: AreaIndex):
            dist = 1
            return dist if node1 in self._edges.get(node2) \
                else self.INF
        def dijkstra(graph: 'HSectionsGraph', start: AreaIndex, end: AreaIndex):
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

        path, distances = dijkstra(self, start, target)
        return path


def generate_hsections_graph(surf: np.array) -> HSectionsGraph:
    def extract_area(mat: np.array, i: int, j: int,
                     coord_to_area_idx: CoordToAreaIndex,
                     area_idx: int) -> tuple[set[Coord], int]:
        # if (i,j) is already part of an area, return empty set and unchanged area_idx
        if coord_to_area_idx.get((i, j)) is not None:
            return set(), area_idx

        q = [(i, j)]

        area = []

        while len(q):
            # print(used, '\n', i, j)
            i, j = q.pop(-1)
            # add to area and mark coords as used:
            area.append((i, j))
            coord_to_area_idx[(i, j)] = area_idx

            for i2, j2 in grid_neighbors_gen(i, j, *GRID_SIZE):
                if coord_to_area_idx.get((i2, j2)) is None and \
                        math.isclose(mat[i, j], mat[i2, j2]):
                    q.append((i2, j2))
        return set(area), area_idx + 1

    def extract_contour(a: set[Coord],
                        a_idx: int,
                        maxi: int, maxj: int,
                        coord_to_area_idx: CoordToAreaIndex,
                        shared_contour: dict[Coord, set[Coord]]):
        # TODO Ed, apply the 0-dim1 0-dim2 rule using a separate function :)
        contour = set()

        for pt in a:
            is_contour_pt = False
            for pt2 in grid_neighbors_gen(*pt, maxi=maxi, maxj=maxj,
                                          include_corner_neighbors=True):
                if pt2 not in a:
                    is_contour_pt = True
                    a2_idx = coord_to_area_idx[pt2]
                    if not shared_contour.get((a_idx, a2_idx)):
                        shared_contour[(a_idx, a2_idx)] = set()
                    shared_contour[(a_idx, a2_idx)].add(pt)
                    if not shared_contour.get((a2_idx, a_idx)):
                        shared_contour[(a2_idx, a_idx)] = set()
                    shared_contour[(a2_idx, a_idx)].add(pt2)
            if is_contour_pt:
                contour.add(pt)
        return contour

    GRID_SIZE = surf.shape
    # computed height sections:
    hsections = []
    coord_to_area_idx: CoordToAreaIndex = {}
    shared_contour: dict[HSectionIndexPair, set[
        Coord]] = {}  # mapping of type    { (0,1) : {(),(),()}  meaning contour of "Sect 0", adjacent to "Sect 1"
    # extract areas and contours
    curr_area_idx = 0
    # first extract areas
    areas = []
    for i, j in itertools.product(range(GRID_SIZE[0]),
                                  range(GRID_SIZE[1])):
        area, curr_area_idx = extract_area(surf, i, j,
                                           coord_to_area_idx=coord_to_area_idx,
                                           area_idx=curr_area_idx)
        if area:
            areas.append(area)
    # second, extract contour and group with areas
    for area_idx, area in enumerate(areas):
        contour = extract_contour(a=area, a_idx=area_idx,
                                  maxi=GRID_SIZE[0], maxj=GRID_SIZE[1],
                                  coord_to_area_idx=coord_to_area_idx,
                                  shared_contour=shared_contour)
        # store area as an HSection
        hsections.append(HSection(area, contour))

    # create graph
    num_sections = len(hsections)
    graph = HSectionsGraph(nodes=range(num_sections),  # the indexes
                           hsections=hsections,
                           shared_contour=shared_contour,
                           coord_to_area_idx=coord_to_area_idx)
    # add edges to the graph, based on shared_contour
    for sect_idx_1, sect_idx_2 in shared_contour.keys():
        graph.add_edge(sect_idx_1, sect_idx_2)

    return graph
