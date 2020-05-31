import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class Complex:
    def __init__(self, terrain, critical_points, persistence):
        self.terrain = terrain
        self.maxima, self.minima, self.saddles, self.monkeys = critical_points
        self.cells = []
        self.persistence_level = persistence

    def _is_smoothened(self, persistence_pairs):
        if len(persistence_pairs) == 0:
            return False
        for i in persistence_pairs:
            if i[1] < self.persistence_level:
                return False
        return True

    def steepest_ascent_trajectory(self, saddle):
        ascending_components = self.get_connected_components(saddle)
        trajectories = []
        for side in ascending_components:
            trajectory = [np.array(saddle)]
            point = saddle
            surrounding = self.get_surrounding(point[0], point[1])
            steepest_ascent = (1, 1)
            for coordinate in side:
                if surrounding[coordinate] > surrounding[steepest_ascent]:
                    steepest_ascent = coordinate
            trajectory.append(np.array(point) + np.array(steepest_ascent) + np.array([-1, -1]))
            point = tuple(trajectory[-1])
            while point not in self.maxima:
                if self.terrain[point] == math.inf:
                    trajectory.append(np.array((0, 0)))
                    point = tuple(trajectory[-1])
                    break
                surrounding = self.get_surrounding(point[0], point[1])
                steepest_ascent = np.unravel_index(np.argmax(surrounding, axis=None), surrounding.shape)
                trajectory.append(np.array(point) + np.array(steepest_ascent) + np.array([-1, -1]))
                point = tuple(trajectory[-1])
            trajectories.append((point, trajectory))
        return trajectories

    def steepest_descent_trajectory(self, saddle):
        descending_components = self.get_connected_components(saddle, ascending=False)
        trajectories = []
        for side in descending_components:
            trajectory = [np.array(saddle)]
            point = saddle
            surrounding = self.get_surrounding(point[0], point[1])
            steepest_descent = (1, 1)
            for coordinate in side:
                if surrounding[coordinate] < surrounding[steepest_descent]:
                    steepest_descent = coordinate
            trajectory.append(np.array(point) + np.array(steepest_descent) + np.array([-1, -1]))
            point = tuple(trajectory[-1])
            while point not in self.minima:
                surrounding = self.get_surrounding(point[0], point[1])
                steepest_descent = np.unravel_index(np.argmin(surrounding, axis=None), surrounding.shape)
                trajectory.append(np.array(point) + np.array(steepest_descent) + np.array([-1, -1]))
                point = tuple(trajectory[-1])
            trajectories.append((point, trajectory))
        return trajectories

    def get_connected_components(self, saddle, ascending=True):
        to_check = {(0, 0): [(1, 0), (0, 1)], (0, 1): [(0, 0), (0, 2)], (0, 2): [(0, 1), (1, 2)],
                    (1, 0): [(0, 0), (2, 0)],
                    (1, 2): [(0, 2), (2, 2)], (2, 0): [(1, 0), (2, 1)], (2, 1): [(2, 0), (2, 2)],
                    (2, 2): [(1, 2), (2, 1)]}
        surrounding = self.get_surrounding(saddle[0], saddle[1])
        connected_components = []
        if ascending:
            surrounding = surrounding > surrounding[1][1]
        else:
            surrounding = surrounding < surrounding[1][1]
        for i in range(3):
            for j in range(3):
                present = False
                if i is not 1 or j is not 1:
                    if surrounding[i, j]:
                        for component in connected_components:
                            if (i, j) in component:
                                present = True
                                break
                        if not present:
                            connected_components.append([(i, j)])
                            connected = True
                            while connected:
                                connected = False
                                for element in connected_components[-1]:
                                    for neighbor in to_check[element]:
                                        if surrounding[neighbor]:
                                            if neighbor not in connected_components[-1]:
                                                connected_components[-1].append(neighbor)
                                                connected = True
        return connected_components

    def get_surrounding(self, i, j, include_self=True):
        """
        Retrieves the surrounding of (i,j) in the terrain
        :param i: first coordinate
        :param j: second coordinate
        :param include_self: Boolean whether or not to include self
        :return: the surrounding of point (i,j)
        """
        surrounding = np.copy(self.terrain[i - 1:i + 2, j - 1:j + 2])
        if not include_self:
            surrounding[1, 1] = np.nan
        return surrounding


class AscendingComplex(Complex):
    def __init__(self, terrain, critical_points, persistence):
        super(AscendingComplex, self).__init__(terrain, critical_points, persistence)
        persistence_pairs = []
        while not self._is_smoothened(persistence_pairs):
            persistence_pairs = []
            self.cell_boundaries = []
            for s in self.saddles:
                boundaries = self.steepest_descent_trajectory(saddle=s)
                for boundary in boundaries:
                    self.cell_boundaries.append((s, boundary))
            for boundary in self.cell_boundaries:
                persistence_pairs.append(
                    ((boundary[0], boundary[1][0]), np.abs(terrain[boundary[1][0]] - terrain[boundary[0]])))


class DescendingComplex(Complex):
    def __init__(self, terrain, critical_points, persistence):
        super(DescendingComplex, self).__init__(terrain, critical_points, persistence)
        persistence_pairs = []
        while not self._is_smoothened(persistence_pairs):
            persistence_pairs = []
            self.cell_boundaries = []
            for s in self.saddles:
                boundaries = self.steepest_ascent_trajectory(saddle=s)
                for boundary in boundaries:
                    self.cell_boundaries.append((s, boundary))
            for boundary in self.cell_boundaries:
                persistence_pairs.append(
                    ((boundary[0], boundary[1][0]), np.abs(terrain[boundary[1][0]] - terrain[boundary[0]])))


class MorseSmaleComplex:
    def __init__(self, terrain, scaling=1, persistence=0):

        self._old_terrain = terrain
        self.terrain = gaussian_filter(self.scale_terrain(terrain, scaling), sigma=2)

        self._prepare_terrain()
        print("Terrain Prepared")
        self.critical_points = self.find_critical_points()
        print("Critical Points Extracted")
        self.ascending_complex = AscendingComplex(self.terrain, self.critical_points, persistence)
        print("Ascending Complex Constructed")
        self.descending_complex = DescendingComplex(self.terrain, self.critical_points, persistence)
        print("Descending Complex Constructed")

    def _prepare_terrain(self):
        """
        Adds a boundary of infinity values and makes sure that adjacent values are not equal
        :return: the prepared terrain
        """
        for i in range(len(self.terrain)):
            self.terrain[i, 0] = math.inf
            self.terrain[i, len(self.terrain[i]) - 1] = math.inf
        for j in range(len(self.terrain[0])):
            self.terrain[0, j] = math.inf
            self.terrain[len(self.terrain) - 1, j] = math.inf
        for i in range(1, len(self.terrain) - 1):
            for j in range(1, len(self.terrain[i]) - 1):
                while self.terrain[i, j] in self.get_surrounding(i, j, include_self=False):
                    self.terrain[i, j] = self.terrain[i, j] + 0.01

    @staticmethod
    def scale_terrain(terrain, scaling):
        """
        Scales the terrain by the given scaling factor (please only insert positive integers)
        :param terrain: The terrain to scale
        :param scaling: The scaling factor

        :return: terrain scaled by a factor 1./scaling
        """
        shp = terrain.shape
        new_shp = tuple(math.ceil(i / scaling) + 2 for i in shp)
        new_terrain = np.zeros(shape=new_shp)
        for i in range(1, len(new_terrain) - 1):
            for j in range(1, len(new_terrain[i]) - 1):
                new_terrain[i, j] = np.average(terrain[(i - 1) * scaling: i * scaling, (j - 1) * scaling:j * scaling])
        return new_terrain

    def is_critical(self, i, j):
        to_check = {(0, 0): [(1, 0), (0, 1)], (0, 1): [(0, 0), (0, 2)], (0, 2): [(0, 1), (1, 2)],
                    (1, 0): [(0, 0), (2, 0)],
                    (1, 2): [(0, 2), (2, 2)], (2, 0): [(1, 0), (2, 1)], (2, 1): [(2, 0), (2, 2)],
                    (2, 2): [(1, 2), (2, 1)]}
        surrounding = self.get_surrounding(i, j)
        surrounding = surrounding > surrounding[1, 1]
        false_boundaries = 0
        false_found = False

        for i in range(3):
            for j in range(3):
                if not i is 1 or not j is 1:
                    if not surrounding[i, j]:
                        false_found = True
                        for neighbor in to_check[(i, j)]:
                            if surrounding[neighbor]:
                                false_boundaries += 1
        return false_boundaries, false_found

    def get_surrounding(self, i, j, include_self=True):
        """
        Retrieves the surrounding of (i,j) in the terrain
        :param i: first coordinate
        :param j: second coordinate
        :param include_self: Boolean whether or not to include self
        :return: the surrounding of point (i,j)
        """
        surrounding = np.copy(self.terrain[i - 1:i + 2, j - 1:j + 2])
        if not include_self:
            surrounding[1, 1] = np.nan
        return surrounding

    def find_critical_points(self):
        maxima = []
        minima = []
        saddles = []
        monkeys = []
        for i in range(1, len(self.terrain) - 1):
            for j in range(1, len(self.terrain[i]) - 1):
                boundaries, false_found = self.is_critical(i, j)
                connected_components = boundaries // 2
                if connected_components is 0:
                    if false_found:
                        maxima.append((i, j))
                    else:
                        minima.append((i, j))
                if connected_components is 2:
                    saddles.append((i, j))
                if connected_components > 2:
                    monkeys.append((i, j))
        return maxima, minima, saddles, monkeys

    def plot(self):
        plt.figure(figsize=(100, 10), dpi=100)
        plt.imshow(self.terrain[1:self.terrain.shape[0] - 1, 1: self.terrain.shape[1] - 1])

        for path in self.descending_complex.cell_boundaries:
            x, y = self.reformat_path(path)
            plt.plot(x - 1, y - 1, c="w")
        for path in self.ascending_complex.cell_boundaries:
            x, y = self.reformat_path(path)
            plt.plot(x - 1, y - 1, c="w")

        maxima = self.reformat_extreme_values(self.critical_points[0])
        minima = self.reformat_extreme_values(self.critical_points[1])
        saddles = self.reformat_extreme_values(self.critical_points[2])
        plt.scatter(maxima[1] - 1, maxima[0] - 1, c="red", marker=",")
        plt.scatter(minima[1] - 1, minima[0] - 1, c="blue", marker=",")
        plt.scatter(saddles[1] - 1, saddles[0] - 1, c="yellow", marker=",")
        plt.savefig("ms_smoothed.png", dpi=100)
        plt.show()

    @staticmethod
    def reformat_extreme_values(extreme_values):
        return np.transpose(np.array(extreme_values))

    @staticmethod
    def reformat_path(path):
        trajectory = path[1][1]
        x = []
        y = []
        for point in trajectory:
            if tuple(point) == (0, 0):
                break
            else:
                x.append(point[1])
                y.append(point[0])
        x = np.array(x)
        y = np.array(y)
        return x, y
