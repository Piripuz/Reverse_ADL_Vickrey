import matplotlib.pyplot as plt


import jax.numpy as jnp
from jax import vmap

from vickrey.likelihood.likelihood import total_log_lik

plt.rcParams["text.usetex"] = True


def plot_contour(
    tt,
    t_as,
    par,
    ax=None,
    x_bounds=None,
    y_bounds=None,
):
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

    x_contour = jnp.linspace(*x_bounds, 81)
    y_contour = jnp.linspace(*y_bounds, 80)
    matrix_actual = vmap(vmap(log_lik, (0, None)), (None, 0))(
        x_contour, y_contour
    )
    if not ax:
        ax = plt.gca()
    ax.contour(x_contour, y_contour, matrix_actual, levels=50)
    names = [r"\mu_\beta", r"\mu_\gamma", r"\mu_t", r"\sigma", r"\sigma_t"]
    ax.set_xlabel("$" + names[x_index] + "$")
    ax.set_ylabel("$" + names[y_index] + "$")
    return ax
