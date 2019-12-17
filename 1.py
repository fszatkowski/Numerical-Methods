from dataclasses import dataclass
from typing import *

import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm


""" This script computes potential and electric field magnitude 
for discrete space with multiple fixed-potential electrodes 
for the rest of the space, u is calculated by solving d2u/dx2 + d2u/dy2 = 0 
using finite difference method with neumann conditions on the borders"""


@dataclass
class Electrode:
    x: Tuple[float, float]
    y: Tuple[float, float]
    u: float
    point_indices: List[Tuple[int, int]] = None

    def __post_init__(self):
        self.point_indices = []

    def contains_idx(self, x: int, y: int) -> bool:
        return (x, y) in self.point_indices


@dataclass
class Spacing:
    min: float
    max: float
    n_points: int

    def __post_init__(self):
        self.h = (self.max - self.min) / (self.n_points - 1)


class Space2D:
    def __init__(
        self, x_spacing: Spacing, y_spacing: Spacing, electrodes: List[Electrode]
    ):
        """ Space for which to solve equations """
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing

        self.u = np.zeros((x_spacing.n_points, y_spacing.n_points))
        self.e = None

        self.electrodes = electrodes
        for electrode in self.electrodes:
            for x in range(x_spacing.n_points):
                for y in range(y_spacing.n_points):
                    x_c = x_spacing.h * x
                    y_c = y_spacing.h * y
                    if (
                        electrode.x[0] <= x_c <= electrode.x[1]
                        and electrode.y[0] <= y_c <= electrode.y[1]
                    ):
                        self.u[x, y] = electrode.u
                        electrode.point_indices.append((x, y))

    def __getitem__(self, idx: Tuple[int, int]) -> float:
        return self.u[idx[0], idx[1]]

    def __setitem__(self, key: Tuple[int, int], value: float):
        if not isinstance(key, tuple):
            raise ValueError("Indexing 2d space with invalid key.")
        if key[0] not in range(0, self.u.shape[0]):
            raise ValueError("Index x out of range.")
        if key[1] not in range(0, self.u.shape[1]):
            raise ValueError("Index y out of shape.")
        self.u[key[0], key[1]] = value

    def plot_u(self):
        plt.figure(num=None, figsize=(8, 5), dpi=80, facecolor='w', edgecolor='k')
        fig, ax = plt.subplots()
        ax.set_title('U')
        im = plt.imshow(self.u.T, origin="lower", cmap="seismic")
        cbar = plt.colorbar(im)
        plt.show()

    def plot_e(self):
        e_x, e_y = np.gradient(self.u, edge_order=2)
        e_x, e_y = -e_x, -e_y
        abs_e = np.sqrt(np.power(e_x, 2) + np.power(e_y, 2))
        xx, yy = np.meshgrid(
            np.linspace(
                self.x_spacing.min, self.x_spacing.max  , self.x_spacing.n_points
            ),
            np.linspace(
                self.y_spacing.min, self.y_spacing.max, self.y_spacing.n_points
            ),
        )

        step = 10
        plt.figure(num=None, figsize=(8, 5), dpi=80, facecolor='w', edgecolor='k')
        fig, ax = plt.subplots()
        plt.gca().set_aspect('equal', adjustable='box')
        CS = ax.contour(xx[::step, ::step].T, yy[::step, ::step].T, abs_e[::step, ::step], levels=10)
        ax.clabel(CS, inline=1, fontsize=10)
        ax.set_title('E magnitude')
        plt.show()


class DifferentialEquations:
    def __init__(self, space: Space2D, u: float):
        """
        Equations express U(x, y) for whole space
        u is constant value for condition: d2u/dx2 + d2y/dy2 = u
        """
        self.space = space
        self.u = u
        self.A: np.array = None
        self.b: np.array = None

        self.eq_idx_to_point: np.array
        self.point_to_idx: Dict[Tuple[int, int], int] = {}
        self._generate_eq_idx_to_point_mapping(space=space)
        self._generate_equations()

    def _generate_eq_idx_to_point_mapping(self, space: Space2D):
        equations = []
        eq_idx = 0
        for x_idx in tqdm(
            range(0, space.x_spacing.n_points),
            total=space.x_spacing.n_points,
            desc="Generating equations for nodes",
        ):
            for y_idx in range(0, space.y_spacing.n_points):
                if not any(
                    [
                        electrode.contains_idx(x_idx, y_idx)
                        for electrode in self.space.electrodes
                    ]
                ):
                    equations.append([x_idx, y_idx])
                    self.point_to_idx[(x_idx, y_idx)] = eq_idx
                    eq_idx += 1

        self.eq_idx_to_point = np.asarray(equations)

    def _generate_equations(self):
        """ Generate eqations for points for which second derivative is relevant """
        self.A = np.zeros((len(self.eq_idx_to_point), len(self.eq_idx_to_point)))
        self.b = np.zeros((len(self.eq_idx_to_point), 1))

        for eq_idx in tqdm(
            range(self.eq_idx_to_point.shape[0]),
            desc="Generating equations",
            total=self.eq_idx_to_point.shape[0],
        ):
            value = self.u
            # initialize coefficients
            x_m1_c = 0
            x_p1_c = 0
            y_m1_c = 0
            y_p1_c = 0

            # get indices of nodes
            x, y = self.eq_idx_to_point[eq_idx][0], self.eq_idx_to_point[eq_idx][1]
            x_m1_i = None
            x_p1_i = None
            y_m1_i = None
            y_p1_i = None

            # check for x-1, y
            # if border, add to x+1 modifier (neumman)
            if self.eq_idx_to_point[eq_idx][0] == 0:
                x_p1_c += 1 / (self.space.x_spacing.h * self.space.x_spacing.h)

            else:
                # check if x-1, y is inside electrode
                f_x_m1_e = False
                for electrode in self.space.electrodes:
                    if electrode.contains_idx(x - 1, y):
                        value -= electrode.u / (
                            self.space.x_spacing.h * self.space.x_spacing.h
                        )
                        f_x_m1_e = True
                # if elecrode was not detected, add modifier to x-1
                if not f_x_m1_e:
                    x_m1_i = self.point_to_idx[(x - 1, y)]
                    x_m1_c += 1 / (self.space.x_spacing.h * self.space.x_spacing.h)

            # check for x+1, y
            # if border, add to x-1 modifier (neumman)
            if self.eq_idx_to_point[eq_idx][0] == self.space.x_spacing.n_points - 1:
                x_m1_c += 1 / (self.space.x_spacing.h * self.space.x_spacing.h)

            else:
                # check if x+1, y is inside electrode
                f_x_p1_e = False
                for electrode in self.space.electrodes:
                    if electrode.contains_idx(x + 1, y):
                        value -= electrode.u / (
                            self.space.x_spacing.h * self.space.x_spacing.h
                        )
                        f_x_p1_e = True
                # if elecrode was not detected, add modifier to x-1
                if not f_x_p1_e:
                    x_p1_i = self.point_to_idx[(x + 1, y)]
                    x_p1_c += 1 / (self.space.x_spacing.h * self.space.x_spacing.h)

            # check for x, y-1
            # if border, add to y+1 modifier (neumman)
            if self.eq_idx_to_point[eq_idx][1] == 0:
                y_p1_c += 1 / (self.space.y_spacing.h * self.space.y_spacing.h)

            else:
                # check if y-1, y is inside electrode
                f_y_m1_e = False
                for electrode in self.space.electrodes:
                    if electrode.contains_idx(x, y - 1):
                        value -= electrode.u / (
                            self.space.y_spacing.h * self.space.y_spacing.h
                        )
                        f_y_m1_e = True
                # if elecrode was not detected, add modifier to x-1
                if not f_y_m1_e:
                    y_m1_i = self.point_to_idx[(x, y - 1)]
                    y_m1_c += 1 / (self.space.y_spacing.h * self.space.y_spacing.h)

            # check for x, y+1
            # if border, add to y-1 modifier (neumman)
            if self.eq_idx_to_point[eq_idx][1] == self.space.y_spacing.n_points - 1:
                y_m1_c += 1 / (self.space.y_spacing.h * self.space.y_spacing.h)

            else:
                # check if y-1, y is inside electrode
                f_y_p1_e = False
                for electrode in self.space.electrodes:
                    if electrode.contains_idx(x, y + 1):
                        value -= electrode.u / (
                            self.space.y_spacing.h * self.space.y_spacing.h
                        )
                        f_y_p1_e = True
                # if elecrode was not detected, add modifier to x-1
                if not f_y_p1_e:
                    y_p1_i = self.point_to_idx[(x, y + 1)]
                    y_p1_c += 1 / (self.space.x_spacing.h * self.space.x_spacing.h)

            self.A[eq_idx, eq_idx] = -2 / (
                self.space.x_spacing.h * self.space.x_spacing.h
            ) - 2 / (self.space.y_spacing.h * self.space.y_spacing.h)

            if x_m1_i is not None:
                self.A[eq_idx, x_m1_i] = x_m1_c
            if x_p1_i is not None:
                self.A[eq_idx, x_p1_i] = x_p1_c
            if y_m1_i is not None:
                self.A[eq_idx, y_m1_i] = y_m1_c
            if y_p1_i is not None:
                self.A[eq_idx, y_p1_i] = y_p1_c

            self.b[eq_idx, 0] = value

    def solve(self) -> Space2D:
        u = np.linalg.solve(self.A, self.b)

        for i in range(self.eq_idx_to_point.shape[0]):
            self.space[self.eq_idx_to_point[i][0], self.eq_idx_to_point[i][1]] = u[i, 0]

        return self.space


if __name__ == "__main__":
    space = Space2D(
        x_spacing=Spacing(0, 5, 51),
        y_spacing=Spacing(0, 8, 81),
        electrodes=[Electrode((2, 3), (0, 2), 0), Electrode((3, 4), (5, 8), 10)],
    )
    space.plot_u()

    equations = DifferentialEquations(space, u=0)
    potential = equations.solve()
    potential.plot_u()

    potential.plot_e()