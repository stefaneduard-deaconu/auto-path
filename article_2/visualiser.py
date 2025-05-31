import math

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from scipy.interpolate import splprep, splev

from areas.utils import create_subplots, create_3d_subplots, set_axes_equal
from areas.utils.interpolate import direction, radius, eucl
from main import Experiment

matplotlib.use('TkAgg')

SCALE = 10  # TODO Stefan, can this be removed?
figsize = (12, 8)
figsize_elevation_profile = (24, 8)


def plot_horizontal_curves(path3d: np.array,
                           fig: Figure,
                           ax: Axes):
    def generate_curve_sects(pts: np.array):
        dir1 = direction(*pts[:3])
        start = 0
        mid = 2
        while mid < len(pts) - 1:
            dir2 = direction(*pts[mid - 1:mid + 2])
            if dir2 == dir1:
                pass
            else:
                yield pts[start:mid + 1], 'green' if dir1 < 0 else 'red'
                start = mid
                dir1 = dir2
                mid = start + 2
            mid += 1
        if start < mid + 1:
            yield pts[start:mid + 1], 'green' if dir1 < 0 else 'red'

    def find_smallest_circle(pts: np.array) -> tuple[float, tuple[int, int]]:
        def circle_origins(a: np.array, b: np.array, radius: float) -> tuple[int, int]:
            x1, y1 = a
            x2, y2 = b
            q = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            y3 = (y1 + y2) / 2
            x3 = (x1 + x2) / 2

            # One answer will be:
            x_1 = x3 + np.sqrt(radius ** 2 - (q / 2) ** 2) * (y1 - y2) / q
            y_1 = y3 + np.sqrt(radius ** 2 - (q / 2) ** 2) * (x2 - x1) / q

            # The other will be:
            x_2 = x3 - np.sqrt(radius ** 2 - (q / 2) ** 2) * (y1 - y2) / q
            y_2 = y3 - np.sqrt(radius ** 2 - (q / 2) ** 2) * (x2 - x1) / q

            return (x_1, y_1), (x_2, y_2)

        def circumcenter(p1, p2, p3, radius):
            x1y1, x2y2 = circle_origins(p1, p3, radius)
            x1, y1 = x1y1
            x2, y2 = x2y2

            o1 = np.array([x1, y1])
            o2 = np.array([x2, y2])

            if direction(p1, p2, p3) * direction(p1, o1, p3) < 0:
                o = o1
            else:
                o = o2
            return o

        min_r, center = float('inf'), None
        for p1, p2, p3 in zip(pts, pts[1:], pts[2:]):
            r = radius(p1, p2, p3)
            if r < min_r:
                min_r = r
                center = circumcenter(p1, p2, p3, radius=r)

        return min_r, center

    pts = path3d[:, :2]

    for sect, color in generate_curve_sects(pts):
        min_r, center = find_smallest_circle(sect)
        ax.plot(*zip(*sect), 'green', lw=4)
        if min_r < 100:
            ax.add_patch(Circle(center, radius=min_r,
                                fill=False, color='darkorange', lw=3))
            ax.text(center[0] - 1.8, center[1] - 0.5, '%dm' % (min_r * SCALE), fontdict={'size': 9,
                                                                                         'weight': 'bold'})
            # TODO Ed, trag
    start, target = pts[0], pts[-1]
    ax.scatter([start[0]], [start[1]], c='green', lw=5)
    ax.text(start[0], start[1] - 3.5, ' start', fontdict={'size': 12, 'color': 'green', 'weight': 'bold'})
    ax.scatter([target[0]], [target[1]], c='green', lw=5)
    ax.text(target[0], target[1] + 2, 'target', fontdict={'size': 12, 'color': 'green', 'weight': 'bold'})
    # plt.plot(*zip(*path3d[:, :2]))
    # TODO Ed, was the first variant correct?

    ax.axis('equal')
    # plt.show()
    # plt.savefig('Figure_77.svg')


def plot_inclination(path3d: np.array,
                     fig: Figure,
                     ax: Axes):
    """plot heights"""
    path3d = np.array(path3d)
    x, y, h = zip(*path3d)

    # ax.plot(x, h)
    # ax.axis('equal')
    # # plt.show()

    # ax.set_title('The Inclination (%) of the Smoothed Path')
    ax.set_xlabel('X - Distance (m)')
    ax.set_ylabel('Y - Altitude (m)')
    # set y lims
    ymin = min(h)
    ymax = max(h)
    size = ymax - ymin + 1
    ymin -= 2 * size
    ymax += 2 * size
    ax.set_ylim((ymin, ymax))
    num_yticks = 15
    ydiff = math.floor((ymax - ymin + 1) / num_yticks)
    if ydiff == 0:
        ydiff = 1
    yticks = range(math.floor(ymin), math.ceil(ymax) + 1, ydiff)
    ax.set_yticks(yticks)

    dist = [0] + [eucl(a, b) * SCALE
                  for a, b in zip(path3d[0:, :2],
                                  path3d[1:, :2])]
    dist2 = [0] * len(h)
    for i in range(1, len(h)):
        dist2[i] = dist2[i - 1] + dist[i]

    def generate_sections(x: np.array, y: np.array):
        dim = len(x)
        dh1 = y[1] - y[0]
        i1 = 0
        i2 = 1
        while i2 < dim - 1:
            dh2 = y[i2 + 1] - y[i2]
            if dh2 * dh1 < 0:  # opposite signs
                yield np.array([(x[i], h[i])
                                for i in range(i1, i2 + 1)]), 'green' if dh1 > 0 else 'red'
                dh1 = dh2
                i1 = i2
            i2 += 1
        # TODO Ed, do anything?

    sections = list(generate_sections(dist2, h))
    for pts, color in sections:
        # print(segment_length(pts))
        ax.plot(*zip(*pts), color, lw=2)
        text_coord = (pts[0] + pts[-1]) / 2
        text_coord[1] = max(h) + 0
        dd, dh = pts[-1] - pts[0]
        inclination = math.ceil(abs(dh / dd * 100))
        # TODO Ed, compute the maximal tangent value along the subpath
        text_size = size * .4
        if color == 'green':
            txt = ax.text(text_coord[0] - 5 * text_size, text_coord[1] + 1 * text_size, f'{inclination}%',
                          fontdict={"size": 12})
        elif color == 'red':
            txt = ax.text(text_coord[0] - 8 * text_size, text_coord[1] - 3.5 * text_size, f'-{inclination}%',
                          fontdict={"size": 12})
        # print()

    return sections

    # ax.axis('equal')


class Visualiser:
    def __init__(self, e: Experiment):
        self.area_section = e.area_sections
        self.area = e.area

    def visualise_terrain(self):
        fig, ax = create_3d_subplots(1, 1, figsize=(6, 6))
        ax.set_title('Terrain as a 3D Grid')  # TODO Ed, can you use inclination instead of height for colormap?
        self.area.plot_terrain_3d(fig=fig, ax=ax, noshow=True, horizontal_ratio=1)
        set_axes_equal(ax)
        fig.tight_layout()
        # plt.show()
        plt.savefig('Figure_1_terrain_3d.svg')

    def visualise_radii(self, path: np.array):
        fig, ax = create_subplots(1, 1, figsize=figsize)
        ax.set_title('Figure . - Horizontal curves')
        plot_horizontal_curves(path, fig=fig, ax=ax)
        plt.show()

    def visualise_elevation_profile(self, path2d: np.array):
        path3d = np.array(self.area_section.interpolate_path_height(path2d))

        # TODO Stefan, move this to right place
        # function to compute 2D elevation profile from 3D path
        def compute_elevation_profile(path3d: np.array) -> np.array:
            distances = [0] * len(path3d)
            for i, (pt1, pt2) in enumerate(zip(path3d, path3d[1:])):
                distances[i + 1] = distances[i] + eucl(pt1[:2], pt2[:2]) * SCALE
            return np.array([
                (distances[i], z)  # MAX_HEIGHT_DIFF)
                for i, (x, y, z) in enumerate(path3d)
            ])

        # TODO Stefan, is this method still needed
        # plot_inclination(path3d, fig=fig, ax=ax)

        # ax.axis('equal')
        # TODO update original function.
        # TODO Stefan: Can you also make a 3D interpolation, instead of one horizontal + one vertical
        def interpolate_2d_path_v2(patb: np.array, multiplier: int = 10, s: float = 1) -> np.array:
            """
            Interpolate a 2D path using B-spline interpolation for smooth roads.
            :param path2d: List of (x, y) coordinates
            :param multiplier: Number of points to generate per input point
            :return: Smooth interpolated list of (x, y) coordinates
            """
            if len(patb) < 2:
                return patb  # Not enough points to interpolate

            dimensions = zip(*patb)
            tck, u = splprep([*dimensions], s=s)  # s=0 for interpolation (exact fit)

            u_new = np.linspace(0, 1, len(patb) * multiplier)
            new_dimensions = splev(u_new, tck)

            smooth_path = np.array(list(zip(*new_dimensions)))
            return smooth_path

        actual_road_rough_elevation = compute_elevation_profile(path3d)

        loosely_interpolated_path = interpolate_2d_path_v2(path3d, multiplier=4, s=100)
        actual_road_may_be = compute_elevation_profile(loosely_interpolated_path)

        # plt.plot(*zip(*actual_road_may_be), color='black', linestyle='-', linewidth=1.5, alpha=0.9)
        def get_elevation_profile_sections(path2d: np.array) -> list[np.array]:
            sections: list[list] = [list(path2d[:2])]
            for pt1, pt2, pt3 in zip(path2d, path2d[1:], path2d[2:]):
                _, h1 = pt1
                _, h2 = pt2
                _, h3 = pt3
                if h1 <= h2 <= h3 or h1 >= h2 >= h3:
                    # add to old one
                    sections[-1].append(pt3)
                else:
                    # start new one
                    sections.append([pt1, pt2])
            return [
                np.array(section)
                for section in sections
            ]

        fig, ax = create_subplots(1, 1, figsize=figsize_elevation_profile)
        ax.plot(actual_road_rough_elevation[:, 0], actual_road_rough_elevation[:, 1], linewidth=1.5, c='black',
                alpha=0.6)

        maximum_road_h = max(actual_road_rough_elevation[:, 1])
        for section in get_elevation_profile_sections(actual_road_may_be):
            d = section[-1][1] - section[0][1]
            color = 'red' if d < 0 else 'green'

            maximum = max([
                abs((h2 - h1) / (x2 - x1) ) * 100
                for (x1, h1), (x2, h2) in zip(section, section[1:])
            ])
            inclination = round(-maximum if d < 0 else maximum, 2)
            inclination_average = abs(round(
                (section[-1][1] - section[0][1]) / (section[-1][0] - section[0][0]) * 100,
                2
            ))

            g = sum([np.array(item) for item in section]) / len(section)
            min_h = min((pt[1] for pt in section))
            g = g[0], g[1] - 2.5
            if d >= 0:
                g = g[0], maximum_road_h + 1.5
            else:
                g = g[0], maximum_road_h + 0.5

            t = plt.text(*g, f'{'↑' if d >= 0 else '↓'} {inclination_average}%',
                         fontdict={"size": 14, "weight": "bold", 'color': color},
                         ha='center', va='center')
            # ax.axis('equal')
            plt.xlabel('Distance from Start (meters)', fontsize=14, color='black')
            plt.ylabel('Road Elevation (meters)', fontsize=14, color='black')

            plt.plot(
                [x for x, h in section],
                [h for x, h in section],
                color=color, linestyle='-', linewidth=2.5, alpha=0.9
            )

        ax.set_title('Elevation Profile', fontdict={"size": 24})
        ax.grid('equal')
        ax.yaxis.grid(True, linestyle='--', linewidth=1, color='gray', alpha=0.3)
        ax.xaxis.grid(True, linestyle='--', linewidth=1, color='gray', alpha=0.3)
        # ax.yaxis.grid(False)

        yticks = list(range(
            math.floor(min(actual_road_rough_elevation[:, 1])),
            math.ceil(max(actual_road_rough_elevation[:, 1])) + 1 + 4,
        ))
        xticks = list(range(
            math.floor(min(actual_road_rough_elevation[:, 0])),
            math.ceil(max(actual_road_rough_elevation[:, 0])) + 1,
            100
        ))
        print(yticks)
        plt.yticks(yticks, fontsize=12)
        plt.xticks(xticks, fontsize=12)
        plt.savefig("Figure_elevation_profile_sections.svg")
        # plt.show()
