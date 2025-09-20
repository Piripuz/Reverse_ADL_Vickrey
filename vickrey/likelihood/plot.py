"""Functions to make plots about the likelihood."""

import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp
from jax import vmap

from vickrey.likelihood.likelihood import total_liks, total_log_lik

plt.rcParams["text.usetex"] = True


def plot_contour(
    tt,
    t_as,
    par,
    ax=None,
    x_bounds=None,
    y_bounds=None,
):
    """Plot a 2-dimensional slice of the likelihood function.

    Args:
        tt: Instance of the TravelTime class, containing the travel
            time function to plot the likelihood for.
        t_as: Vector of arrival time to plot the likelihood of.
        par: Parameters with which the likelihood will be plotted.
            Two entries must be None: the slice will be plotted
            using the other values, for the dimensions the None values
            are in.
        ax: matplotlib.pyplot.Axes instance, on which the contour will
            be plotted. If no axis is given, the current axis will be
            used.
        x_bounds, y_bounds: Bounds of the dimensions that will be
            plotted. If no value is given, the bounds will be
            automatically determined from a list of predefined bounds.

    Returns:
        contour: The contour, which has been plotted on axis ax.
    """
    indices = [i for i, v in enumerate(par) if not v]
    if len(indices) != 2:
        raise ValueError(
            f"The parameter list must have two None values.\n\
            {par} has instead {len(indices)}"
        )
    x_index, y_index = indices
    bounds = [(0.01, 0.6), (0.01, 0.6), (6.5, 11.0), (0.01, 0.6), (0.1, 2.0)]
    if not x_bounds:
        x_bounds = bounds[x_index]
    if not y_bounds:
        y_bounds = bounds[y_index]

    def log_lik(x, y):
        new_par = par.copy()
        new_par[x_index] = x
        new_par[y_index] = y
        return total_log_lik(tt, t_as)(*new_par)

    x_contour = jnp.linspace(*x_bounds, 51)
    y_contour = jnp.linspace(*y_bounds, 50)
    matrix_actual = vmap(vmap(log_lik, (0, None)), (None, 0))(
        x_contour, y_contour
    )
    if not ax:
        ax = plt.gca()
    contour = ax.contour(x_contour, y_contour, matrix_actual, levels=50)
    names = [r"\mu_\beta", r"\mu_\gamma", r"\mu_t", r"\sigma", r"\sigma_t"]
    ax.set_xlabel("$" + names[x_index] + "$")
    ax.set_ylabel("$" + names[y_index] + "$")
    return contour
