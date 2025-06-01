import os

import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, CubicHermiteSpline

from areas.utils.interpolate import Coord3D, is_collinear, distance_to_line
from article_2.visibility_calculator import VisibilityCalculator
from article_2.visualiser import Visualiser
from main import *

import matplotlib
from data.configs import article_config, fast_configs, over_100_seeds_for_algorithm1

matplotlib.use('TkAgg')


# TODO Stefan, temporarily rework this function, then move it back to .interpolate
def remove_bad_points(path3d: list[Coord3D], minimal_radius=15, GRID_RATIO_TO_METERS=10):
    bad_points = {
        'collinear': [],
        'almost_collinear': [],
        'too_small_radius': []
    }

    # Step 1. remove collinear points

    path = [path3d[0],
            *[p2
              for p1, p2, p3 in zip(path3d[0:],
                                    path3d[1:],
                                    path3d[2:])
              if not is_collinear(p1, p2, p3)],
            path3d[-1]]
    bad_points['collinear'] = [p2
                               for p1, p2, p3 in zip(path3d[0:],
                                                     path3d[1:],
                                                     path3d[2:])
                               if is_collinear(p1, p2, p3)]

    # Step 2. remove point who are outside the minimum radius
    #         of two consecutive lines

    min_radius = minimal_radius / GRID_RATIO_TO_METERS  # 15m, but each element on the grid has 5 meters
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
                    new_path[start],
                    new_path[len(new_path) - 1]))  # TODO Ed, error: may exceed if at the end of the path
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
                bad_points['almost_collinear'].extend(new_path[start + 1:end])
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
                bad_points['almost_collinear'].append(new_path[i + 1])
                new_path[i + 1] = None
            else:
                # long before, short after, is the same as line 147 (first else from the for)
                # extract longest possible line
                start, end = extract_longest_line(i,
                                                  new_path)  # TODO may sometime unite a few short lines, with a long line
                # line is from path[start] to path[end+1], so we remote path[start+1:end+1]
                bad_points['almost_collinear'].extend(new_path[start + 1:end])
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


import numpy as np
from scipy.interpolate import splprep, splev
from typing import List, Tuple

Coord = Tuple[float, float]


def interpolate_2d_path_v2(path2d: list[Coord], multiplier: int = 10, s_value: float = 10) -> list[Coord]:
    """
    Interpolate a 2D path using B-spline interpolation for smooth roads.
    :param path2d: List of (x, y) coordinates
    :param multiplier: Number of points to generate per input point
    :return: Smooth interpolated list of (x, y) coordinates
    """
    if len(path2d) < 2:
        return path2d  # Not enough points to interpolate

    dimensions = zip(*path2d)
    tck, u = splprep([*dimensions], s=s_value)  # s=0 for interpolation (exact fit)

    u_new = np.linspace(0, 1, len(path2d) * multiplier)
    new_dimensions = splev(u_new, tck)

    smooth_path = np.array(list(zip(*new_dimensions)))
    return smooth_path


from main import DIJKSTRA_FUNCTIONS


# TODO Stefan 1: run on over 100 seeds
# TODO important functions:
def generate_and_cache(config: TerrainGeneratorConfig, generate_road_based_on: DIJKSTRA_FUNCTIONS = None, ):
    with timer("generate experiment"):
        e = Experiment(config=config)
        e.generate(cache=True)
        e.reset_objective(
            (6, 6),
            (config.GRID_SIZE[0] - 6, config.GRID_SIZE[1] - 6)
        )
        if generate_road_based_on is not None:
            return e, np.array(
                e.test_dijkstra_variants(cache=True, noshow=True, only_compute_based_on=generate_road_based_on)[
                    'height']
            )
        else:
            return e, None


def algorithm_1(rough_path: np.array, MINIMAL_RADIUS: float = 15, GRID_RATIO_TO_METERS: float = 10):
    print(rough_path)
    (
        representative_path,
        removed_from_rough_path
    ) = remove_bad_points(rough_path, minimal_radius=MINIMAL_RADIUS)

    from pprint import pprint
    print(' >> representative points')
    pprint(representative_path)
    print(' >> non-representative points')
    pprint(removed_from_rough_path)

    # scatter rough path
    # plt.plot(rough_path[:, 0], rough_path[:, 1], color='black', label='rough')

    # scatter non-representative points
    set_from_ndarray = lambda arr: set((x, y) for x, y in arr)
    non_representative_points = set_from_ndarray(removed_from_rough_path['almost_collinear']) | \
                                set_from_ndarray(removed_from_rough_path['too_small_radius']) | \
                                set_from_ndarray(removed_from_rough_path['collinear'])

    non_representative_path = np.array([
        pt for pt in rough_path
        if tuple(pt) in non_representative_points
    ])
    # print(list(representative_path))
    # print(list(non_representative_path))
    # plt.scatter(non_representative_path[:, 0],
    #             non_representative_path[:, 1],
    #             s=[.05 * len(non_representative_path)],
    #             color='red',
    #             label='non-representative')

    # scatter representative points

    # TODO STEFAN, DO THIS FOR COLINEAR PATHS :)
    new_rep_path = [representative_path[0]]
    MIN_RADIUS = 15
    for pt1, pt2 in zip(representative_path[1:], representative_path[2:]):
        new_rep_path.append(pt1)
        # optionally append a middle point
        distance = np.linalg.norm(pt2 - pt1)
        if distance * GRID_RATIO_TO_METERS > MIN_RADIUS * 3.1458:
            mid_pt = (pt1 + pt2) / 2
            new_rep_path.append(mid_pt)
            print(pt1, pt2, mid_pt)
    new_rep_path.append(representative_path[-1])
    new_rep_path = np.array(new_rep_path)
    # TODO Stefan, I updated this
    # plt.scatter(new_rep_path[:, 0],
    #             new_rep_path[:, 1],
    #             s=[2 * len(new_rep_path)],
    #             color='green',
    #             label='representative')

    # scatter interpolated path
    # interpolated_path = interpolate_2d_path_v2(representative_path, multiplier=8)

    # plt.plot(interpolated_path[:, 0],
    #          interpolated_path[:, 1],
    #          linewidth=1.5,
    #          color='blue',
    #          label='interpolated')
    #
    # plt.show()
    return interpolate_2d_path_v2(new_rep_path, multiplier=8, s_value=5), (
        new_rep_path
    )


if __name__ == '__main__':
    items = []
    for config in over_100_seeds_for_algorithm1:
        items.append(
            generate_and_cache(config, generate_road_based_on='height')
        )
    # TODO 1. function for horizontal interpolation based on algorithm 1 (remove points and interpolate
    prefix = './figures/'
    try:
        os.mkdir(prefix)
    except:
        pass
    for i, (e, dijkstra_road) in enumerate(items):
        experiment_number = str(i).zfill(3)
        experiment_prefix = prefix +experiment_number
        viz = Visualiser(e)
        # plot terrain and dijkstra raw result based on height
        fig, ax = viz.visualise_terrain_2d(dijkstra_road)
        fig.savefig(f'{experiment_prefix}_1_terrain_and_dijkstra.png', bbox_inches='tight',
                    pad_inches=0)
        plt.close(fig)

        # plot horizontal curves after barebone algorithm 1
        MIN_RADIUS = 15
        algorithm_1_path, (rep_path) = algorithm_1(dijkstra_road, MINIMAL_RADIUS=MIN_RADIUS, GRID_RATIO_TO_METERS=10)
        fig, ax = viz.visualise_radii(algorithm_1_path)
        fig.savefig(f'{experiment_prefix}_3_radii_after_algorithm_1.png', bbox_inches='tight',
                    pad_inches=0)
        plt.close(fig)

        # dijkstra_road = np.array([(10, 10), (10, 11), (11, 11), (12, 11)])  # TODO delete
        fig, ax = viz.visualise_radii(dijkstra_road)
        fig.savefig(f'{experiment_prefix}_3_radii_before_algorithm_1.png', bbox_inches='tight',
                    pad_inches=0)
        plt.close(fig)

        # plot smooth road on terrain, after algorithm 1
        # algorithm_1_path = algorithm_1(dijkstra_road, MINIMAL_RADIUS=30)
        fig, ax = viz.visualise_terrain_2d(algorithm_1_path)
        ax.plot(*zip(*rep_path), color='blue', linestyle='--', linewidth=3, alpha=0.6)
        ax.scatter(*zip(*rep_path), s=[50] * len(rep_path), color='blue', linestyle='--', alpha=0.7)
        fig.savefig(f'{experiment_prefix}_2_terrain_and_path_after_algorithm_1.png', bbox_inches='tight',
                    pad_inches=0)
        plt.close(fig)

        # plot elevation profile, after algorithm 2 (the profiling interpolation :) )
        fig, ax = viz.visualise_elevation_profile(algorithm_1_path,
                                                  s_value=150)
        fig.savefig(f'{experiment_prefix}_4_elevation_profiles_after_algorithms_1_and_2.png', bbox_inches='tight',
                    pad_inches=0)
        plt.close(fig)

        from scipy.spatial.distance import directed_hausdorff

        d1 = directed_hausdorff(dijkstra_road, algorithm_1_path)[0]
        d2 = directed_hausdorff(algorithm_1_path, dijkstra_road)[0]

        hausdorff_distance = max(d1, d2)
        print('distance:', experiment_number, hausdorff_distance)
        print('done')
    # TODO 2. compare Dijkstra data with smoothed data :) ---> radii, height sections (both inclination and visibility)

    quit(0)

if __name__ == '__main__':
    # 1. Choose an area and plot it
    config = fast_configs[-4]
    config = fast_configs[-5]
    # TODO Ed, set the square size
    with timer("generate experiment"):
        e = Experiment(config=config)
        e.generate(cache=True)
        e.reset_objective(
            (5, 5),
            (config.GRID_SIZE[0] - 5, config.GRID_SIZE[1] - 5)
        )
        rough_path = np.array(
            e.test_dijkstra_variants(cache=True, noshow=True, only_compute_based_on='height')['height']
        )

    # rough vs smooth:
    if False:
        plt.plot(rough_path[:, 0], rough_path[:, 1], color='gray', label='rough', linestyle='--')
        plt.plot(interpolated_path[:, 0],
                 interpolated_path[:, 1],
                 linewidth=2,
                 color='blue',
                 label='interpolated')
        plt.grid()
        plt.show()

    viz = Visualiser(e)
    viz.visualise_terrain_2d(interpolated_path)
    # viz.visualise_terrain_3d()
    # viz.visualise_radii(interpolated_path)

    # TODO Stefan, only for elevation:
    dijkstra_road, elevation = viz.visualise_elevation_profile(interpolated_path)
    dijkstra_road *= (10, 10, 1)
    # vis_calculator = VisibilityCalculator(road, observer_height=1.1)
    # vis_calculator.calculate_visibility()
