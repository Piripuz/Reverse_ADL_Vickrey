import jax.numpy as jnp
from jax import vmap, jit
from jaxopt import GradientDescent
from scipy.optimize import minimize

from vickrey.likelihood.likelihood import total_log_lik

def grid_search(
        t_as,
        tt,
        num_mu_beta=6,
        num_mu_gamma=6,
        num_mu_t=3,
        num_sigma=4,
        num_sigma_t=3
):
    g_betas = jnp.linspace(.01, tt.maxb, num_mu_beta)
    g_gammas = jnp.linspace(.01, tt.maxg, num_mu_gamma)
    g_ts = jnp.linspace(6, 11, num_mu_t)
    g_sigmas = jnp.linspace(.1, .5, num_sigma)
    g_sigmats = jnp.linspace(1, 4, num_sigma_t)

    mesh_par = jnp.meshgrid(g_betas, g_gammas, g_ts, g_sigmas, g_sigmats)

    vec_lik = vmap(total_log_lik(tt, t_as), (1, 1, 1, 1, 1))
    grid_result = vec_lik(*mesh_par)
    
    best = jnp.array(
        jnp.unravel_index(
            grid_result.argmax(), grid_result.shape
        )
    )
    init = jnp.r_[
        g_betas[best[0]],
        g_gammas[best[1]],
        g_ts[best[2]],
        g_sigmas[best[3]],
        g_sigmats[best[4]]
    ]
    return init

def grad_free(
        t_as,
        tt,
        init,
        verbose=False
):
    @jit
    def lik_fun(par):
        log_lik = total_log_lik(tt, t_as)(*par)
        return -log_lik

    def obj_fun(par):
        print(("{:8.4f}"*5).format(*par))
        return lik_fun(par)

    if verbose:
        res = minimize(obj_fun, init, method="Nelder-Mead")
    else:
        res = minimize(lik_fun, init, method="Nelder-Mead")

    return res
