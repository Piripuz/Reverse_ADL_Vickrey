import jax.numpy as jnp
from jax import vmap

from jaxopt import GradientDescent

from scipy.stats import truncnorm
from scipy.stats import norm

import numpy as np


def cost(travel_time):
    def inner_cost(t_a, beta, gamma, t_star):
        return (
            travel_time.f(t_a)
            + beta * jnp.maximum(0, t_star - t_a)
            + gamma * jnp.maximum(0, t_a - t_star)
        )

    return inner_cost


def find_td(travel_time):
    """Given a travel time, returns a function that, given vectors of
    betas, gammas and t*, returns a vector of the optimal departure
    times.

    """

    def inner_find_td(beta, gamma, t_star):
        cost_fun = cost(travel_time)
        solver = GradientDescent(
            fun=cost_fun, acceleration=False, maxiter=4000, stepsize=0.05
        )
        lval, _ = solver.run(0.0, beta, gamma, t_star)
        rval, _ = solver.run(24.0, beta, gamma, t_star)
        val = jnp.where(
            cost_fun(rval, beta, gamma, t_star)
            < cost_fun(lval, beta, gamma, t_star),
            rval,
            lval,
        )
        return jnp.where(
            cost_fun(val, beta, gamma, t_star)
            < cost_fun(t_star, beta, gamma, t_star),
            val,
            t_star,
        )

    return inner_find_td


def generate_arrival(
    n,
    travel_time,
    mu_beta=0.7,
    mu_gamma=1.2,
    mu_t=9.5,
    sigma=0.1,
    sigma_t=1,
    full_output=False,
    seed=None,
):
    """Generate samples of departure time.

    Arguments:
        n: number of samples
        mu_beta, mu_gamma, mu_t: mean for the parameters beta, gamma and t*
        sigma: variance for the parameters beta and gamma
        sigma_t: variance for the parameter t*

    Returns a numpy array with the data
    """
    random_gen = np.random.RandomState(seed)
    # Betas, gammas and t_star are generated according to the chosen distributions

    betas = truncnorm.rvs(
        -mu_beta / sigma,
        10000,
        loc=mu_beta,
        scale=sigma,
        size=n,
        random_state=random_gen,
    )
    gammas = truncnorm.rvs(
        -mu_gamma / sigma,
        10000,
        loc=mu_gamma,
        scale=sigma,
        size=n,
        random_state=random_gen,
    )
    ts = norm.rvs(mu_t, sigma_t, n, random_state=random_gen)
    t_as = vmap(find_td(travel_time))(betas, gammas, ts)
    if full_output:
        return t_as, betas, gammas, ts
    return t_as
