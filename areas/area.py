import random
import math

# import scientific tools
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

# import perlin noise library
from perlin_numpy import (
    generate_perlin_noise_2d, generate_fractal_noise_2d
)

from auto_path.areas.graph import WeightedGraph
from auto_path.areas.utils import set_axes_equal

import itertools

from auto_path.areas.utils.interpolate import Coord3D


# TODO move to pybezier library
def get_bezier_points(pts: np.ndarray, order: int, time: float) -> np.array:
    if order == 0:
        return np.copy(pts[0])
    # for each two points, find the middle point according to current *time*
    new_pts = np.array([
        (pts[i] * (1 - time) + pts[i + 1] * time)
        for i in range(len(pts) - 1)
    ])

    return get_bezier_points(new_pts, order - 1, time)


def eucl(a: np.array, b: np.array):
    a = np.array(a)
    b = np.array(b)
    return math.sqrt(np.sum((a - b) ** 2))


def segment_length(pts: np.array) -> float:
    return sum([eucl(a, b)
                for a, b in zip(pts, pts[1:])])


def compute_bezier_curve(pts: np.array, spacing=.1) -> np.array:
    points = []
    num_points = int(
        np.sqrt(abs(
            np.sum((pts[0] - pts[-1]) ** 2)
        )) / spacing + 2
    )
    for time in np.linspace(0, 1, num_points):
        points.append(get_bezier_points(pts, pts.shape[0] - 1, time))
    return np.array(points)


class Area:
    def __init__(self, surf: np.array,
                 start: np.array = None, target: np.array = None,
                 min_height: float = None, max_height: float = None):
        self.surf = surf
        if start is None or target is None:
            self.start, self.target = None, None
        else:
            self.start, self.target = [(int(x), int(y))
                                       for x, y in [start, target]]
        self.min_height = min_height
        self.max_height = max_height
        self.GRID_SIZE = self.surf.shape

    @classmethod
    def from_perlin_noise(cls, seed,
                          GRID_SIZE: tuple[int,int],
                          scaling_argument: tuple[int, int],
                          height_interval: tuple[int,int]):
        """
        Summary

        By default, we generate a 2km by 2km grid constructed out of 5m squares

        So the grid size is 200 x 200.
        """
        min_height, max_height = height_interval
        # generate surface
        np.random.seed(seed)
        height_interval = max_height - min_height
        surf = generate_perlin_noise_2d(GRID_SIZE, scaling_argument) * height_interval + min_height  # TODO Ed, 4,4 ?
        # also generate a random objective to start with
        area = cls(surf, min_height=min_height, max_height=max_height)
        area.generate_objective()
        return area

    @property
    def pts3d(self) -> list[Coord3D]:
        return [(x,y,self.surf[x,y])
                for x,y in itertools.product(range(self.dim1),
                                             range(self.dim2))]

    @property
    def shape(self):
        return np.array(self.surf.shape)

    @property
    def dim1(self):
        return self.surf.shape[0]

    @property
    def dim2(self):
        return self.surf.shape[1]

    def set_objective(self, start: np.array, target: np.array):
        self.start, self.target = [(int(x), int(y))
                                   for x, y in [start, target]]

    def generate_objective(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        self.start, self.target = [(int(x), int(y))
                                   for x, y in np.random.rand(2, 2) * self.GRID_SIZE]

    def _plot_surf(self, surf: np.array, flip: bool = True):
        surf = surf.T if flip else surf
        plt.imshow(surf, cmap='Blues', interpolation='lanczos')
        plt.colorbar()
        plt.scatter(*zip(*[self.start, self.target]), c=['red'])

    def plot_terrain(self):
        self._plot_surf(self.surf)

    def show(self):
        plt.show()

    def interpolate_height(self, coord: np.array):
        # interpolate using the 4 closest points
        # TODO Ed, may later use a more mathematic interpolation algo
        # TODO Ed, also upgrade how it's computed
        pt = np.array([math.floor(coord[0]),
                       math.floor(coord[1])])
        closest_pts = [(0, 0), (0, 1), (1, 0), (1, 1)] + pt
        # x,y = zip(*closest_pts, pt)
        # z = [self.surf[i,j]
        #      for i,j in zip(x,y)]
        g = sum([
            self.surf[i, j]
            for i, j in closest_pts
        ]) / len(closest_pts)
        # ax = plt.figure().add_subplot(projection='3d')
        # ax.scatter(x,y,z)
        # ax.scatter([coord[0]], [coord[1]], [g])
        # plt.show()
        return g    

    @classmethod
    def _plot_surf_3d(cls, z: np.array,
                      colormap: str = 'gist_earth',
                      xs: np.array = None,
                      ys: np.array = None,
                      fig: Figure = None,
                      ax: Axes3D = None,
                      axis: str = '',
                      use_tk: bool = True,
                      alpha: float = 1,
                      flip=True):
        """
        TODO Also return the ax,fig,surf :)
        :param surf:
        :param xs:
        :param ys:
        :return:
        """
        # if flip:
        #     z = z.T
        cls._set_backend(use_tk=use_tk)
        if xs is None:
            xs = np.arange(len(z[0]))
        if ys is None:
            ys = np.arange(len(z))

        x, y = np.meshgrid(xs, ys)

        from matplotlib.ticker import LinearLocator

        # create the figure and the 3D axes TODO rewrite projection='3d'
        if not fig or not ax:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # configure axis
        if axis == 'equal':
            ax.set_aspect('equal')
        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=colormap,
                               linewidth=0, antialiased=False,
                               alpha=alpha)
        # # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # # A StrMethodFormatter is used automatically
        # ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        """
        im = ax.imshow(surf, cmap='Blues', interpolation='lanczos')  # TODO Ed, IT DOES INTERPOLATE
        # create custom colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.15)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        """
        fig.colorbar(surf, fraction=.05)

        # After plotting all data:
        if axis == 'equal':
            set_axes_equal(ax)

        return ax, fig, surf

    def plot_vicinity(self,
                      pt: np.array,
                      radius: int,
                      axis: str = '',
                      use_tk: bool = False):  # TODO Ed, can you add sth for each method call, like a context? but based on arguments, not calls?
        self._set_backend(use_tk=use_tk)
        # plot surface
        x1, x2 = pt[0] - radius, pt[0] + radius + 1
        y1, y2 = pt[1] - radius, pt[1] + radius + 1
        x = np.arange(x1, x2)
        y = np.arange(y1, y2)
        fig, ax, surf = Area._plot_surf_3d(z=self.surf[x1:x2, y1:y2],
                                           xs=x,
                                           ys=y)
        px, py = pt
        pz = self.surf[px, py]
        ax.scatter([px], [py], [pz])
        plt.show()

    def plot_terrain_3d(self,
                        ax: Axes3D = None,
                        fig: Figure = None,
                        axis: str = '',
                        noshow: bool = False,
                        use_tk: bool = True):
        ax, fig, surf = Area._plot_surf_3d(self.surf,
                                           colormap="Blues",
                                           ax=ax,
                                           fig=fig,
                                           axis=axis,
                                           use_tk=use_tk)
        if not noshow:
            self.show()
        return ax, fig, surf

    # TODO Ed, task 2, implement interpolation using B-Spline libraries
    #         also:
    #         a) use it in point interpolation method @ AreaSections
    #         b) use 2d interpolation for computing a Bezier Curve from a series of points

    # TODO Ed, current task:
    #  add graph traversal for the original surface

    def generate_graph(self) -> WeightedGraph:
        self.full_graph = WeightedGraph.from_surface(self.surf)
        return self.full_graph

    @classmethod
    def _set_backend(cls, use_tk: bool = False):
        import matplotlib
        if use_tk:
            matplotlib.use('TkAgg')
        else:
            matplotlib.use('module://backend_interagg')
        print(f"Using matplotlib backend={matplotlib.get_backend()}")


# def interpolate_3d_line(area: Area,
#                         start: tuple[int, int],
#                         target: tuple[int, int]) -> list[tuple[int, int]]:
#     line_length = segment_length(np.array([start, target]))
#     line = []
#     a, b = np.array(start), np.array(target)  # TODO Ed, can be np.array([start, target])
#     for i in np.linspace(0, 1, math.ceil(line_length)):
#         coord = a * (1 - i) + b
#         h = area.interpolate_height(coord=coord)
#         line.append([*coord, h])
#     return np.array(line)


def interpolate_using_bezier_curve(area: Area,
                                   waypoints: np.array) -> list[tuple[int, int]]:
    waypoints = np.array(waypoints)
    area.plot_terrain()
    plt.plot(*zip(*waypoints), 'orange')
    line = []
    # step 1: add the height to waypoints
    for coord in waypoints:
        h = area.interpolate_height(coord=coord)
        line.append([*coord, h])
    waypoints = np.array(line)

    # step2: compute the Bezier Curve
    curve = compute_bezier_curve(waypoints)

    plt.plot(*zip(*curve[:, :2]), 'limegreen')
    plt.show()
    return curve
