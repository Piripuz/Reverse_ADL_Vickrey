import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import vmap

from vickrey.likelihood.likelihood import total_log_lik

def plot_contour(
        tt,
        t_as,
        ax=None,
        x_name="mu_beta",
        y_name="mu_gamma",
        x_bounds=[.01, .99],
        y_bounds=[.01, 3.],
        par=[None, None, 9.5, .1, 1.5]
):
    names = ["mu_beta", "mu_gamma", "mu_star", "sigma", "sigma_star"]
    if x_name not in names:
        raise ValueError(f"The value {x_name} was not recognized as a parameter name")
    if y_name not in names:
        raise ValueError(f"The value {y_name} was not recognized as a parameter name")
    def log_lik(x, y):
        x_index, y_index = names.index(x_name), names.index(y_name)
        par[x_index] = x
        par[y_index] = y
        return total_log_lik(tt, t_as)(*par)
    x_contour = jnp.linspace(*x_bounds, 201)
    y_contour = jnp.linspace(*y_bounds, 200)
    m_contour = jnp.meshgrid(x_contour, y_contour)
    matrix_actual = vmap(vmap(log_lik, (0, None)), (None, 0))(x_contour, y_contour)
    if not ax:
        ax = plt.gca()
    ax.contour(x_contour, y_contour, matrix_actual, levels=50)
    return ax
