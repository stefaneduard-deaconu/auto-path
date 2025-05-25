import math

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from scipy.interpolate import splprep, splev

from areas.utils import create_subplots
from areas.utils.interpolate import direction, radius, eucl
from main import Experiment

matplotlib.use('TkAgg')

SCALE = 10  # TODO Stefan, can this be removed?
figsize = (12, 8)


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

    def visualise_radii(self, path: np.array):
        fig, ax = create_subplots(1, 1, figsize=figsize)
        ax.set_title('Figure . - Horizontal curves')
        plot_horizontal_curves(path, fig=fig, ax=ax)
        plt.show()

    def visualise_elevation_profile(self, path2d: np.array):
        path3d = np.array(self.area_section.interpolate_path_height(path2d))
        fig, ax = create_subplots(1, 1, figsize=figsize)
        ax.set_title('Figure . - Elevation Profile')
        plot_inclination(path3d, fig=fig, ax=ax)

        # ax.axis('equal')
        # TODO separate method: 3d interpolate
        def interpolate_2d_path_v2(patb: np.array, multiplier: int = 10) -> np.array:
            """
            Interpolate a 2D path using B-spline interpolation for smooth roads.
            :param path2d: List of (x, y) coordinates
            :param multiplier: Number of points to generate per input point
            :return: Smooth interpolated list of (x, y) coordinates
            """
            if len(patb) < 2:
                return patb  # Not enough points to interpolate

            dimensions = zip(*patb)
            tck, u = splprep([*dimensions], s=100)  # s=0 for interpolation (exact fit)

            u_new = np.linspace(0, 1, len(patb) * multiplier)
            new_dimensions = splev(u_new, tck)

            smooth_path = np.array(list(zip(*new_dimensions)))
            return smooth_path

        actual_road_may_be = interpolate_2d_path_v2(path3d, multiplier=4)
        distances = [0] * len(actual_road_may_be)
        for i, (pt1, pt2) in enumerate(zip(actual_road_may_be, actual_road_may_be[1:])):
            distances[i + 1] = distances[i] + eucl(pt1[:2], pt2[:2]) * SCALE
        MAX_HEIGHT_DIFF = max(path3d[:, 2]) - min(path3d[:, 2])
        actual_road_may_be = [
            (distances[i], z - 2)  # MAX_HEIGHT_DIFF)
            for i, (x, y, z) in enumerate(actual_road_may_be)
        ]

        # plt.plot(*zip(*actual_road_may_be), color='black', linestyle='-', linewidth=1.5, alpha=0.9)
        sections: list[list] = [actual_road_may_be[:2]]
        for pt1, pt2, pt3 in zip(actual_road_may_be, actual_road_may_be[1:], actual_road_may_be[2:]):
            x1, h1 = pt1
            x2, h2 = pt2
            x3, h3 = pt3
            d1 = h2 - h1
            d2 = h3 - h2
            if d1 * d2 <= 0:
                # start new one
                sections.append([pt1, pt2])
            else:
                # add to old one
                sections[-1].append(pt3)
        for section in sections:
            d = section[1][1] - section[0][1]
            color = 'red' if d > 0 else 'green'

            maximum = max([
                abs((h2 - h1) / (x2 - x1) * 100)
                for (x1, h1), (x2, h2) in zip(section, section[1:])
            ])
            inclination = round(-maximum if d > 0 else maximum, 2)

            g = sum([np.array(item) for item in section]) / len(section)
            g -= (-5, +3)
            plt.text(*g, f'{inclination} %' )

            plt.plot(
                [x for x, h in section],
                [h - 2 for x, h in section],
                color=color, linestyle='-', linewidth=1.5, alpha=0.9
            )
        plt.show()
