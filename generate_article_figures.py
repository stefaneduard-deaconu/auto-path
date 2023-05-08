from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from areas.area import segment_length
from areas.utils import set_axes_equal, create_subplots
from areas.utils.interpolate import remove_bad_points, path_length, interpolate_2d_path_as_is, direction
from main import *

import math

import matplotlib

# # TODO Ed, cannot use in headless mode:
# matplotlib.use('TkAgg')

SCALE = 10

if __name__ == '__main__':
    # 1. Choose an area and plot it
    config = TerrainGeneratorConfig(seed=0, GRID_SIZE=(100, 100),
                                    scaling_argument=(4, 4), height_interval=(100, 120),
                                    height_delta=3)
    # TODO Ed, set the square size
    with timer("generate experiment"):
        e = Experiment(config=config)
        e.generate(cache=True)
        start, target = (5, 5), (config.GRID_SIZE[0] - 5, config.GRID_SIZE[1] - 5)
        e.reset_objective(start, target)
        a = e.area
        as_ = e.area_sections

    # TODO set to True to plot all figures
    plot_all = False
    plot = {"Figure_1": False,
            "Figure_2": False,  # TODO you should flip this, or the surface, but this will take time to find which
            "Figure_3": False,
            "Figure_4": False,
            "Figure_5": False,
            "Figure_6": False,
            "Figure_7": False,
            "Figure_8": False,
            "Figure_9": True,
            "Figure_10": True}
    breakpoint()
    # TODO Figure 1. Terrain (3D grid)
    if plot_all or plot['Figure_1']:
        fig, ax = create_3d_subplots(1, 1, figsize=(6, 6))
        ax.set_title('Terrain as a 3D Grid')  # TODO Ed, can you use inclination instead of height for colormap?
        a.plot_terrain_3d(fig=fig, ax=ax, noshow=True)
        set_axes_equal(ax)
        fig.tight_layout()
        plt.show()

    # OLD Figure 2. Plot the area using triangulation
    # # TODO Ed, not working
    #
    # # (2)
    # fig, (ax1, ax2) = create_3d_subplots(1, 2, figsize=(12, 6))
    # ax1.set_title('Terrain as a Delaunay triangulation')  # TODO Ed, can you use inclination instead of height for colormap?
    # ax2.set_title('Terrain as the 3D Grid that resulted from triangulation')  # TODO Ed, can you use inclination instead of height for colormap?
    #
    # import numpy as np
    # from scipy.spatial import Delaunay
    # # Create a Delaunay triangulation from the points
    #
    # grid = a.surf[:10, :10]
    # grid_pts = [(x,y,z)
    #             for x,y,z in a.pts3d
    #             if x < 10 and y < 10]
    # sparse_pts = np.array([p
    #                        for p in grid_pts
    #                        if np.random.random() < .10])  # keep 20% of the points
    # tri = Delaunay(sparse_pts)
    # # # Print the simplices (triangles) in the triangulation
    # # print(tri.simplices)
    # # plot pts
    # ax1.scatter(*zip(*sparse_pts))
    # # Plot the triangles
    # triangles = Poly3DCollection(sparse_pts[tri.simplices])
    # triangles.set_alpha(0.2)
    # triangles.set_facecolor('b')
    # ax1.add_collection(triangles)
    # # a.plot_terrain_3d(fig=fig, ax=ax, noshow=True)
    #
    # # plot real surface on ax2:
    # Area._plot_surf_3d(grid, ax=ax2, fig=fig)
    #
    # set_axes_equal(ax1)
    # set_axes_equal(ax2)
    # fig.tight_layout()
    # plt.show()

    # # TODO Ed, this is too difficult and maybe useless
    # #  3. Plot a path (will use Dijkstra later)
    # # NICE, could use ML to choose which points to keep, by taking into account an angle to all prev/future points of less than n meters etc

    if plot_all or plot['Figure_2']:
        # TODO Figure 2 - plot the selected sections
        fig, ax = create_subplots(1, 1, figsize=(10, 10))
        e.area_sections.plot_selected_sections(noshow=True,
                                               ax=ax, fig=fig)
        plt.show()

    # TODO Figure 3 - plotting the Dijkstra path
    breakpoint()
    paths = e.test_dijkstra_variants(cache=True, noshow=True)
    path_height = paths['height']
    if plot_all or plot['Figure_3']:
        # # TODO Ed, 3D
        # fig, ax = create_3d_subplots(1, 1)
        # e.area_sections.plot_path(path_height, ax=ax, fig=fig)
        # set_axes_equal(ax)  ## TODO Ed, use this insid the show() function
        # e.area_sections.show()

        # TODO Ed, 2D
        #  will be moved to another code area
        fig, ax = create_subplots(1, 1)
        e.area_sections.plot_path_2d(path_height, ax=ax, fig=fig)
        ax.axis('equal')  ## TODO Ed, use this insid the show() function
        e.area_sections.show()

    # OLD    4. extract essential points from path
    #        5. Smooth (interpolate) with cubic spline
    # fig, (ax1, ax2, ax3) = create_3d_subplots(1, 2)
    # fig, (ax1, ax2) = create_3d_subplots(1, 2, figsize=(10, 6))
    # ax1.set_title("Original 3D Grid Path")
    # ax2.set_title("3D Grid Path after only important points are kept")
    # ax1.plot(*zip(*path), 'red', marker='o', markersize=3)
    # ax2.plot(*zip(*path_rough), 'red', marker='o', markersize=3)
    # plt.show()

    # TODO Figure 4-5  smoothing and interpolation
    path_rough1, bad_points = remove_bad_points(path_height, minimal_radius=25)
    path_smooth1 = interpolate_2d_path_as_is(path_rough1, multiplier=2)

    path_rough2, _ = remove_bad_points(path_smooth1, minimal_radius=25)
    path_smooth2 = interpolate_2d_path_as_is(path_rough2, multiplier=4)
    if plot_all or plot['Figure_6']:
        # Steps 0-2, 3
        fig, (ax1, ax2) = create_subplots(1, 2)
        # ax1: initial points, without colinear ones (scattered as red crosses
        ax1.scatter(*zip(*path_height), c='green', marker='o', lw=5, s=20)
        ax1.scatter(*zip(*bad_points['colinear']), c='red', marker='x', lw=1, s=20)
        ax1.scatter(*zip(*bad_points['almost_colinear']), c='orange', marker='+', lw=1, s=55)
        ax1.axis('equal')
        ax1.set_xlabel('(a) - Steps 0-2')
        ax1.grid()

        ax2.plot(*zip(*path_smooth1), 'orange', marker='o', markersize=4)
        ax2.scatter(*zip(*path_rough1), c='green', marker='o', lw=5, s=20)
        ax2.axis('equal')
        ax2.set_xlabel('(b) - Step 3. Cubic B-Spline Interpolation')
        ax2.grid()

        #     ax.grid()

        # # Algorithm applied for 2 iterations
        # fig, (ax1, ax2) = create_subplots(1, 2)
        # # ax1: initial points, without colinear ones (scattered as red crosses
        # ax1.plot(*zip(*path_smooth1), c='green', marker='o')
        # ax2.plot(*zip(*path_smooth2), c='green', marker='o')
        #
        # ax1.axis('equal')
        # ax1.set_xlabel('Iteration 1')
        # ax1.grid()
        #
        # ax2.axis('equal')
        # ax2.set_xlabel('Iteration 2')
        # ax2.grid()

        plt.show()


    # TODO Ed, add these?
    # # 5.extra Rectify the radii which are too tight
    def radius(p1: np.array,
               p2: np.array,
               p3: np.array) -> float:
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        # Compute the distances between the points
        a = np.linalg.norm(p1 - p2)
        b = np.linalg.norm(p2 - p3)
        c = np.linalg.norm(p3 - p1)
        # Compute the semi-perimeter of the triangle
        s = (a + b + c) / 2
        # Compute the area of the triangle using Heron's formula
        A = np.sqrt(s * (s - a) * (s - b) * (s - c))
        # Compute the circumradius of the triangle
        R = (a * b * c) / (4 * A)
        return R


    #
    #
    # # fig, ax = create_3d_subplots(1, 1)
    # path3d = [(*coord, e.area_sections.interpolate_height(coord))
    #           for coord in path_smooth]
    # c = [float('inf'),
    #      *[radius(*path3d[i - 1:i + 2]) * 10
    #        for i in range(1, len(path3d) - 1)],
    #      float('inf')]
    # c[0] = c[-1] = max(c[1:-1]) + 1
    # # rectification: find close points with radius < 15,
    # #  remove them, after which we'll replace with the points on a circle or radius similar to the neighboring points
    # min_radius = 15
    # min_diameter = 30
    #
    #
    # def generate_group(idxs: list[int],
    #                    pts: list[Coord]):
    #     i1 = 0
    #     i2 = 1
    #     while i2 < len(idxs):
    #         if eucl(pts[i1], pts[i2]) <= min_diameter * 1:
    #             i2 += 1
    #         yield idxs[i1], idxs[i2 - 1]
    #         i1, i2 = i2, i2 + 1
    #     if i2 < len(idxs):
    #         yield idxs[i1], idxs[i2]  # only if they're different?
    #     elif i1 < len(idxs):
    #         yield (idxs[i1],)
    #
    #
    def weight_center(pts: np.array):
        pts = np.array(pts)
        dim = len(pts)
        mean_x, mean_y, mean_z = sum(pts[:, 0]) / dim, \
                                 sum(pts[:, 1]) / dim, \
                                 sum(pts[:, 2]) / dim
        return np.array([mean_x, mean_y, mean_z])


    #
    #
    # to_remove = [i
    #              for i in range(len(c))
    #              if c[i] < 15]
    # to_remove_groups = None  # TODO Ed, would be better the check the similarity of that circle, to the points, or find the right circle
    # to_remove_groups = list(generate_group(to_remove, path3d))
    #
    # spacing = path_length(path3d) / (len(path3d) - 1)
    # for lt, rt in reversed(to_remove_groups):
    #     lt, rt = lt - 1, rt + 1
    #     replaced_path = path3d[lt - 1:rt + 1]
    #     replace_path = [path3d[lt],
    #                     weight_center(path3d[lt:rt + 1]),
    #                     path3d[rt]]
    #     replace_path_smoothed_2d = interpolate_2d_path_as_is([p[:2]
    #                                                           for p in replace_path],
    #                                                          multiplier=2)
    #     replace_path_smoothed = e.area_sections.interpolate_path_height(replace_path_smoothed_2d)
    #     # add the replacement
    #     # path3d[lt:rt+1] = replace_path_smoothed
    #     path3d = path3d[:lt] + replace_path_smoothed + path3d[rt + 1:]
    #     # TODO Ed, also smooth
    # # path_smoother = interpolate_2d_path([(x, y) for x, y, z in path3d[lt - 1:rt + 1 + 1]],
    # #                                     multiplier=2)
    # # path3d_smoother = [(*coord2d, e.area_sections.interpolate_height(coord2d))
    # #                    for coord2d in path_smoother]
    # path3d_smoother = path3d
    #
    # c2 = [float('inf'),
    #       *[radius(*path3d[i - 1:i + 2]) * 10
    #         for i in range(1, len(path3d) - 1)],
    #       float('inf')]
    # c2[0] = c2[-1] = max(c2[1:-1]) + 1
    #
    # # for i, (x, y, z) in enumerate(path3d):
    # #     ax2.text(x, y, z, f'{c[i]}')
    # sc = ax2.plot(*zip(*path3d_smoother), 'green')
    # set_axes_equal(ax1)
    # set_axes_equal(ax2)
    # plt.show()

    # TODO Ed, use for final Figures
    path3d = path_smooth2  # TODO Ed, what to do with this?

    pass  # TODO final results


    #       6. Plot data about the road:
    #           a. horizontal curves
    #           b. inclination
    #           c. 3d?

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
                ax.text(center[0] - 1.8, center[1] - 0.5, '%dm' % (min_r*SCALE), fontdict={'size': 9,
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
        plt.show()


    def plot_inclination(path3d: np.array,
                         fig: Figure,
                         ax: Axes):
        """plot heights"""
        path3d = np.array(path3d)
        x, y, h = zip(*path3d)

        # ax.plot(x, h)
        # ax.axis('equal')
        # plt.show()

        ax.set_title('The Inclination (%) of the Smoothed Path')
        ax.set_xlabel('X - Distance (m)')
        ax.set_ylabel('Y - Altitude (m)')

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

        for pts, color in generate_sections(dist2, h):
            print(segment_length(pts))
            ax.plot(*zip(*pts), color, lw=2)
            text_coord = (pts[0] + pts[-1]) / 2
            text_coord[1] = max(h) + 0
            dd, dh = pts[-1] - pts[0]
            inclination = math.ceil(abs(dh / dd * 100))
            if color == 'green':
                txt = ax.text(text_coord[0] - 25, text_coord[1] + 80, f'{inclination}%', fontdict={"size": 12})
            elif color == 'red':
                txt = ax.text(text_coord[0] - 40, text_coord[1] + 10, f'-{inclination}%', fontdict={"size": 12})
            print()

        ax.axis('equal')


    def plot_3d_path(path3d: np.array,
                     fig: Figure,
                     ax: Axes3D):
        e.area_sections.plot_path_3d_real(path3d,
                                          fig=fig,
                                          ax=ax)


    # TODO Figures 6. 7. 8.

    if plot_all or plot['Figure_7']:
        fig, ax = create_subplots(1, 1, figsize=(12, 8))
        ax.set_title('Figure 7 - inclination')
        plot_inclination(path3d, fig=fig, ax=ax)
        ax.axis('equal')
        plt.show()

    if plot_all or plot['Figure_8']:  # TODO !!!!!!!!!
        fig, ax = create_3d_subplots(1, 1, figsize=(12, 8))
        ax.set_title('Figure 8 - 3D Final Path')
        plot_3d_path(path3d, fig=fig, ax=ax)
        ax.axis('equal')
        plt.show()

    if plot_all or plot['Figure_9']:  # TODO !!!!!!!!!
        fig, ax = create_subplots(1, 1, figsize=(12, 8))
        ax.set_title('Figure 9 - Horizontal curves')
        plot_horizontal_curves(path3d, fig=fig, ax=ax)
        ax.axis('equal')
        plt.show()
