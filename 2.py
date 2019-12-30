import numpy as np
import tqdm as tqdm


from dataclasses import dataclass
from typing import List, Tuple, Sequence

import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm


""" This script computes temperature across 2d space and time using cranck-nicholson schema
we assume start values are 30 degrees at heaters and zero elsewhere """


@dataclass
class Heater:
    x: Tuple[float, float]
    y: Tuple[float, float]
    t: float
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


class Space2DWithTime:
    def __init__(
        self,
        x_spacing: Spacing,
        y_spacing: Spacing,
        t_spacing: Spacing,
        heaters: Sequence[Heater],
        initial_t: int = 10,
        alpha: float = 1,
    ):
        """ Space for which to solve equations """
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing
        self.t_spacing = t_spacing

        self.heaters = heaters
        self.initial_t = initial_t
        self.t, self.current_t = self.initialize_t()
        self.alpha = alpha

    def __getitem__(self, idx: Tuple[int, int, int]) -> float:
        return self.t[idx[0], idx[1], idx[2]]

    def __setitem__(self, key: Tuple[int, int, int], value: float):
        if not isinstance(key, tuple):
            raise ValueError("Indexing with object.")
        if key[0] not in range(0, self.t.shape[0]):
            raise ValueError("Index x out of range.")
        if key[1] not in range(0, self.t.shape[1]):
            raise ValueError("Index y out of shape.")
        if key[2] not in range(0, self.t.shape[2]):
            raise ValueError("Index t out of shape.")
        self.t[key[0], key[1], key[2]] = value

    def initialize_t(self) -> Tuple[np.array, int]:
        t = (
            np.ones(
                (
                    self.x_spacing.n_points,
                    self.y_spacing.n_points,
                    self.t_spacing.n_points,
                )
            )
            * self.initial_t
        )
        for heater in self.heaters:
            for x in range(self.x_spacing.n_points):
                for y in range(self.y_spacing.n_points):
                    x_c = self.x_spacing.h * x
                    y_c = self.y_spacing.h * y
                    if (
                        heater.x[0] <= x_c <= heater.x[1]
                        and heater.y[0] <= y_c <= heater.y[1]
                    ):
                        t[x, y, :] = heater.t
                        heater.point_indices.append((x, y))
        return t, 1

    def solve_one_time_step(self) -> np.array:
        if self.current_t >= self.t_spacing.n_points - 1:
            raise ValueError(
                f"Solving eqations for time step: {self.current_t} impossible, max time step solvable: {self.t_spacing.n_points - 2}"
            )

        """ Generate equations mapping"""
        eq_idx = 0

        point_to_eq_idx = {}
        eq_idx_to_point = {}
        t = self.current_t + 1

        for x in range(0, self.x_spacing.n_points):
            for y in range(0, self.y_spacing.n_points):
                if not any([heater.contains_idx(x, y) for heater in self.heaters]):
                    eq_idx_to_point[eq_idx] = (x, y, t)
                    point_to_eq_idx[(x, y, t)] = eq_idx
                    eq_idx += 1

        """ Generate equations """
        num_eqs = eq_idx
        A = np.zeros((eq_idx, eq_idx))
        b = np.zeros((eq_idx, 1))

        x_mod = 1 / (2 * self.x_spacing.h * self.x_spacing.h)
        y_mod = 1 / (2 * self.y_spacing.h * self.y_spacing.h)
        t_mod = 1 / (2 * self.t_spacing.h * self.alpha)

        for i in range(num_eqs):
            # coefficients and indices for 4 closests points in t+1
            xp_c = 0
            xn_c = 0
            yp_c = 0
            yn_c = 0

            xp_i = None
            xn_i = None
            yp_i = None
            yn_i = None

            x, y, t = eq_idx_to_point[i]
            value = 0

            if x == 0:
                xp_c -= x_mod
                value += x_mod * self.t[x + 1, y, t - 1]
            else:
                # check if x-1, y is inside heater
                f_x_m1_e = False
                for heater in self.heaters:
                    if heater.contains_idx(x - 1, y):
                        value += 2* heater.t * x_mod
                        f_x_m1_e = True
                # if heater was not detected, add modifier to x-1
                if not f_x_m1_e:
                    xn_i = point_to_eq_idx[(x - 1, y, t)]
                    xn_c -= x_mod
                    value += x_mod * self.t[x - 1, y, t - 1]

            # check for x+1, y
            # if border, add to x-1 modifier (neumman)
            if x == self.x_spacing.n_points - 1:
                xn_c -= x_mod
                value += x_mod * self.t[x - 1, y, t - 1]
            else:
                # check if x+1, y is inside heater
                f_x_p1_e = False
                for heater in self.heaters:
                    if heater.contains_idx(x + 1, y):
                        value += 2* heater.t * x_mod
                        f_x_p1_e = True
                # if heater was not detected, add modifier to x-1
                if not f_x_p1_e:
                    xp_i = point_to_eq_idx[(x + 1, y, t)]
                    xp_c -= x_mod
                    value += x_mod * self.t[x + 1, y, t - 1]

            # check for x, y-1
            # if border, add to y+1 modifier (neumman)
            if y == 0:
                yp_c -= y_mod
                value += y_mod * self.t[x, y + 1, t - 1]
            else:
                # check if y-1, y is inside heater
                f_y_m1_e = False
                for heater in self.heaters:
                    if heater.contains_idx(x, y - 1):
                        value += 2*heater.t * y_mod
                        f_y_m1_e = True
                # if heater was not detected, add modifier to y-1
                if not f_y_m1_e:
                    yn_i = point_to_eq_idx[(x, y - 1, t)]
                    yn_c -= y_mod
                    value += y_mod * self.t[x, y - 1, t - 1]

            # check for x, y+1
            # if border, add to y-1 modifier (neumman)
            if y == self.y_spacing.n_points - 1:
                yn_c -= y_mod
                value += y_mod * self.t[x, y - 1, t - 1]
            else:
                # check if y+1, y is inside heater
                f_y_p1_e = False
                for heater in self.heaters:
                    if heater.contains_idx(x, y + 1):
                        value += 2*heater.t * y_mod
                        f_y_p1_e = True
                # if heater was not detected, add modifier to x-1
                if not f_y_p1_e:
                    yp_i = point_to_eq_idx[(x, y + 1, t)]
                    yp_c -= y_mod
                    value += y_mod * self.t[x, y + 1, t - 1]

            A[i, i] = t_mod + 2 * (x_mod + y_mod)

            if xp_i is not None:
                A[i, xp_i] = xp_c
            if xn_i is not None:
                A[i, xn_i] = xn_c
            if yp_i is not None:
                A[i, yp_i] = yp_c
            if yn_i is not None:
                A[i, yn_i] = yn_c

            value += (
                - 2 * x_mod * self.t[x, y, t - 1]
                - 2 * y_mod * self.t[x, y, t - 1]
                + t_mod * self.t[x, y, t - 2]
            )
            b[i, 0] = value

        """ Solve equations and return values for t+1 """
        u = np.linalg.solve(A, b)
        for i in range(len(eq_idx_to_point)):
            self.t[eq_idx_to_point[i]] = u[i, 0]
        self.current_t += 1
        return self.t[:, :, t]

    def solve_full(self):
        for _ in range(self.current_t, self.t.shape[2] - 1):
            self.solve_one_time_step()

    def plot(self):
        frames = [self.t[:, :, t].T for t in range(self.t.shape[2])]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        vmax = np.max(frames[0])
        vmin = np.min(frames[0])

        im = ax.imshow(frames[0], origin="lower")
        cb = fig.colorbar(im, cmap="seismic")
        tx = ax.set_title('Temperature at t=0.')

        while True:
            for t, frame in enumerate(frames):
                im.set_data(frame)
                im.set_clim(vmin, vmax)
                tx.set_text(f'Temperature at t={t}.')
                fig.canvas.draw_idle()
                plt.pause(0.01)

    def plot_at_t(self, t:int):
        vmax = np.max(self.t[:,:,0])
        vmin = np.min(self.t[:,:,0])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f"Temperature at {t}")
        im = plt.imshow(self.t[:,:,t].T, origin="lower")
        im.set_clim(vmin, vmax)
        cb = fig.colorbar(im, cmap="seismic")
        plt.show()


if __name__ == "__main__":
    """ To ensure stability alpha and steps for time, x and y dimensions has to be set to satisfy:
    1/4 > alpha * ht / (hx*hy) """
    space = Space2DWithTime(
        x_spacing=Spacing(0, 5, 11),
        y_spacing=Spacing(0, 8, 17),
        t_spacing=Spacing(0, 200, 401),
        heaters=[Heater((2, 3), (0, 2), 30), Heater((3, 4), (5, 8), 30)],
        initial_t=10,
        alpha=0.01
    )
    space.solve_full()
    space.plot()
