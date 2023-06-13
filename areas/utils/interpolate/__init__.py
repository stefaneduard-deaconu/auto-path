import itertools

import numpy as np
from scipy.interpolate import CubicSpline

from areas.graph import Coord
from areas.utils import create_3d_subplots, set_axes_equal

Coord3D = tuple[float, float, float]


def interpolate_3d_surf(sparse_data: list[tuple[int, int, int]],
                        shape: tuple[float, float],
                        method: str = "linear") -> np.array:
    import numpy as np
    from scipy.interpolate import griddata

    # Create some random data for the surface
    x, y, z = zip(*sparse_data)

    # Create a grid of points to interpolate onto
    xi = np.arange(shape[0])
    yi = np.arange(shape[1])
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the surface onto the grid
    zi = griddata((y, x), z, (xi, yi), method=method)  # TODO Ed, why switch x with y? :/

    return zi


sq_size = 10  # 10m for each unit on the graph
import math


def direction(p1: np.array, p2: np.array, p3: np.array):
    return -1 if np.linalg.det([
        [*p1, 1],
        [*p2, 1],
        [*p3, 1],
    ]) < 0 else 1


def is_colinear(p1: np.array, p2: np.array, p3: np.array):
    return abs(
        np.linalg.det([
            [*p1, 1],
            [*p2, 1],
            [*p3, 1],
        ])
    ) < 0.00001


def distance_to_line(point: np.array,
                     line: tuple[np.array]) -> float:
    line = np.array(line)
    line_direction = line[1] - line[0]

    # Compute the vector representing the line segment from line_point_1 to the point
    line_segment = point - line[0]

    # Compute the cross product of the two vectors
    cross_product = np.cross(line_direction, line_segment)

    # Compute the distance from the point to the line
    return np.linalg.norm(cross_product) / np.linalg.norm(line_direction)


def eucl(a: np.array, b: np.array):  # TODO Ed, add to a library
    a = np.array(a)
    b = np.array(b)
    return math.sqrt(np.sum((a - b) ** 2))


def path_length(path: np.array):  # TODO Ed, add to a library
    return sum([eucl(a, b)
                for a, b in zip(path,
                                path[1:])])


def remove_bad_points(path3d: list[Coord3D], minimal_radius=15):
    bad_points = {
        'colinear': [],
        'almost_colinear': [],
        'too_small_radius': []
    }

    # Step 1. remove colinear points

    # fig, (ax1, ax2, ax3) = create_3d_subplots(1, 3, figsize=(12, 7))
    # ax1.set_title('Step 1. remove colinear points')
    # ax2.set_title('Step 2. remove point which are outside the minimal radius')
    # ax1.plot(*zip(*path), c='blue', marker='o', markersize=1)

    # 1. remove colinear points
    path = [path3d[0],
            *[p2
              for p1, p2, p3 in zip(path3d[0:],
                                    path3d[1:],
                                    path3d[2:])
              if not is_colinear(p1, p2, p3)],
            path3d[-1]]
    bad_points['colinear'] = [p2
                              for p1, p2, p3 in zip(path3d[0:],
                                                    path3d[1:],
                                                    path3d[2:])
                              if is_colinear(p1, p2, p3)]

    # Step 2. remove point who are outside the minimum radius
    #         of two consecutive lines

    # def radius(p1: np.array,
    #            p2: np.array,
    #            p3: np.array) -> float:
    #     # Compute the distances between the points
    #     a = np.linalg.norm(p1 - p2)
    #     b = np.linalg.norm(p2 - p3)
    #     c = np.linalg.norm(p3 - p1)
    #
    #     # Compute the semi-perimeter of the triangle
    #     s = (a + b + c) / 2
    #
    #     # Compute the area of the triangle using Heron's formula
    #     A = np.sqrt(s * (s - a) * (s - b) * (s - c))
    #
    #     # Compute the circumradius of the triangle
    #     R = (a * b * c) / (4 * A)
    #     return R
    # min_radius = 15 / 10  # 15m, but each element on the grid has 5 meters
    # p1, p2 = path[:2]
    # s_path = [p1]
    # for p3 in path[2:]:
    #     if radius(p1, p2, p3) < min_radius:
    #         # remove p2
    #         p2 = p3
    #     else:
    #         s_path.append(p2)
    #         p1 = p2
    #         p3 = p3

    # Step 2.
    min_radius = minimal_radius / 10  # 15m, but each element on the grid has 5 meters
    min_diameter = 2 * min_radius

    # 1) check if radius is big enough
    short_lines = {i
                   for i in range(len(path) - 1)
                   if eucl(path[i], path[i + 1]) < min_diameter}
    long_lines = {i
                  for i in range(len(path) - 1)
                  if eucl(path[i], path[i + 1]) >= min_diameter}
    # look for consecutive short lines, and try to merge them
    # they are "mergeable" if the straight line from start to bottom, and all the short lines together,
    #  have a negligable area, as compared with the number of square to traverse.
    new_path = list(path)

    # cases: 1. long - short - long, we ignore
    #           if same directions => raise error
    def is_almost_line(start: int,
                       end: int,
                       new_path: np.array):
        for i in range(start, min(end + 1,
                                  len(new_path))):
            p = new_path[i]
            try:
                d = distance_to_line(p, (new_path[start], new_path[end + 1]))
            except:
                d = distance_to_line(p, (
                new_path[start], new_path[len(new_path) - 1]))  # TODO Ed, error: may exceed if at the end of the path
            if d > min_radius:
                return False  # return False if at least a point is too far from the line
        return True

    def extract_longest_line(i: int, new_path: np.array):
        start = i
        end = i + 1

        while end < len(new_path) and is_almost_line(start, end, new_path) \
                and end not in long_lines:  # TODO Ed, second cond is not needed,
            # TODO Ed, but we need to keep some of the removed pts
            end += 1
        return start, end + 1

    ignore_until = 0
    for i in short_lines:
        if i < ignore_until:
            continue
        if new_path[i] is None:
            continue

        if i - 1 < 0:
            # special cases:
            if i + 1 in long_lines:
                new_path[i + 1] = None
            else:
                # extract longest possible line
                start, end = extract_longest_line(i,
                                                  new_path)  # TODO Ed, we remove everything up until the first long line, is itok?
                # TODO Ed, mostly yes (from prev line), but depends on the min_radius setting
                # line is from path[start] to path[end+1], so we remote path[start+1:end+1]
                bad_points['almost_colinear'].extend(new_path[start+1:end])
                new_path[start + 1:end] = [None] * (end - (start + 1))
                # ignore the point in the big fore
                ignore_until = end
        else:
            # we have both previous and next line

            # if previous is short, you failer
            if new_path[i - 1] is not None and i - 1 in short_lines:
                raise Exception('BAD1')
            # if both are long, we'll ignore the second point from this line
            if i + 1 in long_lines:
                # TODO Ed, instead of this, we should replace with another point on this short line?
                bad_points['almost_colinear'].append(new_path[i + 1])
                new_path[i + 1] = None
            else:
                # long before, short after, is the same as line 147 (first else from the for)
                # extract longest possible line
                start, end = extract_longest_line(i,
                                                  new_path)  # TODO may sometime unite a few short lines, with a long line
                # line is from path[start] to path[end+1], so we remote path[start+1:end+1]
                bad_points['almost_colinear'].extend(new_path[start+1:end])
                new_path[start + 1:end] = [None] * (end - (start + 1))
                # ignore the point in the big fore
                ignore_until = end
    # never remove last point TODO Ed
    if new_path[-1] is None:
        new_path[-1] = path3d[-1]  # TODO Ed, extremely dirty
    # TODO Ed, depending on angle, there's a minimal length for the lines
    return np.array([x
                     for x in new_path
                     if x is not None]), \
           bad_points


def interpolate_2d_path(path2d: list[Coord],
                        multiplier: int = 4):
    # Apply cubic spline interpolation to the real coordinates
    # pts_to_interpolate = remove_bad_points(path2d)
    pts_to_interpolate = path2d

    t = np.arange(len(pts_to_interpolate))
    cs = CubicSpline(t, pts_to_interpolate, bc_type='natural')
    smooth_path = cs(np.linspace(0, len(pts_to_interpolate) - 1, multiplier * len(pts_to_interpolate)))
    return smooth_path


def interpolate_2d_path_as_is(path: list[Coord],
                              multiplier: int = 4):
    # Apply cubic spline interpolation to the real coordinates

    t = np.arange(len(path))
    cs = CubicSpline(t, path, bc_type='natural')
    smooth_path = cs(np.linspace(0, len(path) - 1, multiplier * len(path)))
    return smooth_path


if __name__ == '__main__':
    path = [(0, 0),
            (1,),
            (2,),
            (3,),
            (4,),
            (5,),
            (6,),
            ]

# if __name__ == '__main__':
#     import matplotlib
#     import matplotlib.pyplot as plt
#     from areas.area import Area
#     from matplotlib.figure import Figure
#     from mpl_toolkits.mplot3d import Axes3D
#
#
#     # TODO Ed, use the next one for nice 3d subplots
#     def our_impl():
#         # TEST create a sparse surface and plot it both before and after interpolation
#         matplotlib.use('TkAgg')
#         a = Area.from_perlin_noise(seed=111,
#                                    GRID_SIZE=(50, 50), scaling_argument=(2, 2),
#                                    min_height=100, max_height=115)
#         fig, (ax1, ax2) = create_3d_subplots(1, 2)
#         # config the figures
#         fig.set_figwidth(18)
#         fig.set_figheight(10)
#         # ax1
#         a.plot_terrain_3d(fig=fig,
#                           ax=ax1,
#                           noshow=True,
#                           # axis='equal'  # TODO Ed, same as set_axes_equal(ax1)
#                           )
#         # ax2
#         pts_3d = [(x, y, a.surf[x, y])
#                   for x, y in itertools.product(np.arange(a.dim1),
#                                                 np.arange(a.dim2))]
#         ax2.scatter(*zip(*pts_3d), c='blue', lw=0, s=5, marker='.')
#         # prepare for show:
#         set_axes_equal(ax1)
#         set_axes_equal(ax2)
#         plt.show()
#
#
#     # TODO Ed, use the next one for nice 3d subplots
#     def interpolate_3d(ptd_3d_sparse, shape, method):
#         pass
#
#
#     def sparse_to_grid():
#         # TEST create a sparse surface and plot it both before and after interpolation
#         matplotlib.use('TkAgg')
#         a = Area.from_perlin_noise(seed=111,
#                                    GRID_SIZE=(50, 50), scaling_argument=(2, 2),
#                                    min_height=100, max_height=115)
#         fig, ((ax1, ax2), (ax3, ax4)) = create_3d_subplots(2, 2)
#         # config the figures
#         fig.set_figwidth(9)
#         fig.set_figheight(9)
#         # ax1 - before sparsing
#         pts_3d = [(x, y, a.surf[x, y])
#                   for x, y in itertools.product(np.arange(a.dim1),
#                                                 np.arange(a.dim2))]
#         ax1.scatter(*zip(*pts_3d), c='blue', lw=0, s=9, marker='.')
#         # ax2 - after sparsing
#         ptd_3d_sparse = [pt_3d for pt_3d in pts_3d
#                          if np.random.random() < .3]
#         ax2.scatter(*zip(*ptd_3d_sparse), c='blue', lw=0, s=9, marker='.')
#         # prepare for show:
#
#         # Line 2 - start from sparse non-grid-based data, to obtain a grid based non-sparse data
#         # TODO Ed, should also use the triangulated data
#         # TODO Ed, IMPORTANT, Than triangulated => grid based is actually a interpolation application
#
#         # ax3: plot the grid based surface
#         Area._plot_surf_3d(a.surf,
#                            ax=ax3,
#                            fig=fig)
#         # ax4: plot the interpolated surface
#         surf_interpolated = interpolate_3d(ptd_3d_sparse,
#                                            a.shape,
#                                            method='linear')
#         Area._plot_surf_3d(surf_interpolated,
#                            ax=ax4,
#                            fig=fig)
#         set_axes_equal(ax1)
#         set_axes_equal(ax2)
#         set_axes_equal(ax3)
#         set_axes_equal(ax4)
#         plt.show()
#
#     # our_impl()
#     # sparse_to_grid()
#
#     # TODO Ed, implement interpolation for a 3d path
