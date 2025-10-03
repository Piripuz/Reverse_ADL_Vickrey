"""Runs an optimization cycle and print the result on a contour plot."""

import numpy as np
import matplotlib.pyplot as plt

from vickrey.real_data.fit_function import tt_data
from vickrey.likelihood.generate_data import generate_arrival
from vickrey.likelihood.plot import plot_contour
from vickrey.likelihood.optimize import optim_cycle
from vickrey.utils import is_weekend

plt.rcParams["text.usetex"] = True

num_it = 3

errs = np.zeros((num_it, 5))
days = []
while len(days) < num_it:
    day = np.random.randint(1, 181)
    if not is_weekend(day):
        days.append(day)

for i, day in enumerate(days):
    tt = tt_data(day)

    num = 1000

    filename = "../../img/various_contours/iteration_{}_params_{}_{}.png"

    # Parameters are drawn from a uniform distribution. Bounds are decided
    # based on characteristic of the travel time function. Variance of
    # sigma is limited in order to keep a meaningful (different from
    # uniform) distribution where values of beta, gamma are interesting
    # (that is, lower than tt.maxb, tt.maxg).
    par = np.random.uniform(
        [0.05, 0.05, 7.5, 0.01, 0.05],
        [tt.maxb, tt.maxg, 11, np.minimum(tt.maxb, tt.maxg) / 2, 1.5],
    )
    print(
        "\n".join(
            [
                "Performing estimation for parameters",
                f"{par}",
                "",
            ]
        )
    )
    t_as = generate_arrival(num, tt, *par)
    res = optim_cycle(t_as, tt, par)

    for plot_indices in (np.r_[0, 1], np.r_[3, 4]):
        par_subs = [
            p if i not in plot_indices else None for i, p in enumerate(par)
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_contour(
            tt,
            t_as,
            par_subs,
            ax,
        )
        ax.scatter(
            *np.r_[par][plot_indices],
            color="green",
            label="Correct parameters",
            zorder=2.5,
        )
        ax.scatter(
            *np.r_[res.x][plot_indices],
            color="red",
            label="Retrieved parameters",
            zorder=2.5,
        )
        ax.legend()
        par_colored = [
            f"{p:.2f}"
            if i not in plot_indices
            else r"$\bf{" + f"{p:.2f}" + r"}$"
            for i, p in enumerate(par)
        ]
        ax.set_title(f"Day {day}: " + ", ".join(par_colored))
        fig.savefig(
            filename.format(i, *plot_indices),
            dpi=600,
        )
    print(f"Iteration {i} completed.\n\n")
    errs[i] = np.abs(np.r_[res.x] - par) / par
tot_err = np.mean(errs, axis=0)
print(
    "\n".join(
        [
            "Total error:",
            str(tot_err),
            "",
        ]
    )
)
