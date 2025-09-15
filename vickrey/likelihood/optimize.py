import jax.numpy as jnp
from jax import vmap, jit
from scipy.optimize import minimize
from time import time

from vickrey.likelihood.likelihood import total_log_lik


def grid_search(
    t_as,
    tt,
    num_mu_beta=6,
    num_mu_gamma=6,
    num_mu_t=3,
    num_sigma=4,
    num_sigma_t=5,
):
    g_betas = jnp.linspace(0.01, tt.maxb, num_mu_beta)
    g_gammas = jnp.linspace(0.01, tt.maxg, num_mu_gamma)
    g_ts = jnp.linspace(6, 11, num_mu_t)
    g_sigmas = jnp.linspace(0.1, 0.5, num_sigma)
    g_sigmats = jnp.linspace(0.2, 1.5, num_sigma_t)

    mesh_par = jnp.meshgrid(g_betas, g_gammas, g_ts, g_sigmas, g_sigmats)

    vec_lik = vmap(total_log_lik(tt, t_as), (1, 1, 1, 1, 1))
    grid_result = vec_lik(*mesh_par)

    best = jnp.array(
        jnp.unravel_index(grid_result.argmax(), grid_result.shape)
    )
    init = jnp.r_[
        g_betas[best[0]],
        g_gammas[best[1]],
        g_ts[best[2]],
        g_sigmas[best[3]],
        g_sigmats[best[4]],
    ]
    return init


def grad_free(t_as, tt, init, verbose=False):
    @jit
    def lik_fun(par):
        log_lik = total_log_lik(tt, t_as)(*par)
        return -log_lik

    def obj_fun(par):
        print(("{:8.4f}" * 5).format(*par))
        return lik_fun(par)

    if verbose:
        res = minimize(obj_fun, init, method="Nelder-Mead", tol=1e-3)
    else:
        res = minimize(lik_fun, init, method="Nelder-Mead", tol=1e-3)

    return res


def optim_cycle(t_as, tt, par=None):
    """Do a full optimization cycle: grid search and optimizer.

    Args:
        t_as: Vector of arrival times, in hours.
        tt: Instance of the TravelTime class, containing the travel
            time function for which the optimization is done.
        par (optional): original parameters. If given, the relative
            errors are computed and printed.

    Returns:
        res: final result of the iterative optimizer.
    """
    start = time()
    init = grid_search(t_as, tt)

    print(
        "\n".join(
            [
                f"In {time() - start:.2f} seconds, found initial conditions",
                f"{init}",
                "Starting iterative optimizer...",
                "",
            ]
        )
    )

    start = time()
    res = grad_free(t_as, tt, init)
    if res.status:
        breakpoint()

    print(
        "\n".join(
            [
                f"In {time() - start:.2f} seconds, optimizer converged to",
                f"{res.x}",
            ]
        )
    )
    if par:
        print(
            "\n".join(
                [
                    "Relative errors:",
                    f"{jnp.abs(jnp.r_[res.x] - jnp.r_[par]) / jnp.r_[par]}",
                ]
            )
        )
    return res
