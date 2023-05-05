# TODO move the neighbor generators in here
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


def grid_neighbors_gen(i: int, j: int, maxi: int, maxj: int,
                       include_corner_neighbors: bool = False,
                       filter_by: Callable = None) -> list[tuple[int, int]]:
    coords = [(i - 1, j) if i > 0 else None,
              (i, j - 1) if j > 0 else None,
              (i, j + 1) if j < maxj - 1 else None,
              (i + 1, j) if i < maxi - 1 else None]
    if include_corner_neighbors:
        coords.extend(
            [(i - 1, j - 1) if i > 0 and j > 0 else None,
             (i - 1, j + 1) if i > 0 and j < maxj - 1 else None,
             (i + 1, j + 1) if i < maxi - 1 and j < maxj - 1 else None,
             (i + 1, j - 1) if i < maxi - 1 and j > 0 else None])
    return list(filter(lambda x: x is not None,
                       coords))


def are_grid_neighbors(a: tuple[int, int], b: tuple[int, int],
                       include_corner_neighbors: bool = False):
    dx, dy = a[0] - b[0], a[1] - b[1]
    answer = (abs(dx) <= 1 or abs(dy) <= 1)  # cornesc included
    if include_corner_neighbors:
        return answer
    else:
        return answer and dx * dy == 0


def are_corner_neighbors(a: tuple[int, int], b: tuple[int, int]) -> bool:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return dx * dy == 1


def minmax(values: list, key: Callable = None):
    if not key:
        key = lambda x: x
    minimum = min(values, key=key)
    maximum = max(values, key=key)
    return minimum, maximum


# Plotting functions:


def create_3d_subplots(nrows: int, ncols: int,
                       figsize: tuple[int, int] = None) -> tuple[Figure, Axes3D]:
    # TODO Ed, different call for 1,1
    fig_axes = plt.subplots(nrows, ncols, subplot_kw={'projection': '3d'})
    if figsize:
        fig_axes[0].set_figwidth(figsize[0])
        fig_axes[0].set_figheight(figsize[1])
    return fig_axes
def create_subplots(nrows: int, ncols: int,  # TODO Ed, can be enhanced
                    figsize: tuple[int, int] = None) -> tuple[Figure, Axes3D]:
    # TODO Ed, different call for 1,1
    fig_axes = plt.subplots(nrows, ncols)
    if figsize:
        fig_axes[0].set_figwidth(figsize[0])
        fig_axes[0].set_figheight(figsize[1])
    return fig_axes


def set_axes_equal(ax):
    '''
    ORIGIN: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    TIME: 4/23/2023 17:05 EEST

    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
