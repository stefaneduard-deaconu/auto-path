import numpy as np

import numpy as np

from areas.area import eucl


def __path_distance(path, start_idx, end_idx):
    return sum(eucl(path[i], path[i + 1]) for i in range(start_idx, end_idx))


def __is_visible(path, i, j, observer_height=1.2):
    """Check if P_j is visible from P_i, considering observer eye height."""
    p_i = np.array(path[i], dtype=float)
    p_j = np.array(path[j], dtype=float)

    # Add observer height to z at the starting point
    p_i[2] += observer_height

    for k in range(i + 1, j):
        p_k = np.array(path[k], dtype=float)
        t = (k - i) / (j - i)  # interpolation factor
        interpolated = p_i + t * (p_j - p_i)
        if p_k[2] > interpolated[2]:
            return False
    return True


def compute_min_visibility_distances(path: np.array, observer_height=1.2):
    n = len(path)
    min_visibilities = [0.0] * n

    for i in range(n):
        for j in range(i + 1, n):
            if not __is_visible(path, i, j, observer_height):
                min_visibilities[i] = __path_distance(path, i, j - 1)
                break
        else:
            min_visibilities[i] = __path_distance(path, i, n - 1)

    return min_visibilities




class VisibilityCalculator:
    def __init__(self, elevation_data: np.array, observer_height: float = 1.1):
        self.elevation_data = elevation_data
        self.observer_height = observer_height

    def calculate_visibility(self, override_observer_height: float = None):
        observer_height = override_observer_height or self.observer_height
        visibilities = compute_min_visibility_distances(self.elevation_data, observer_height)
        print(f"Minimal visibility distances (driver eye height = {observer_height}m):")
        for i, v in enumerate(visibilities):
            print(f"From Point {i}: {v:.2f} meters")
        return visibilities

    def calculate_visibility_including_terrain(self, terrain: np.array):
        raise NotImplementedError(f'Future work. feature "{self.__class__}.calculate_visibility_including_terrain(terrain)" NOT IMPLEMENTED!')
