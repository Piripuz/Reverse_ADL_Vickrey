"""Function related to the explicit minimization of the cost."""

import jax.numpy as jnp
from jax import vmap

from jaxopt import GradientDescent

from scipy.stats import truncnorm
from scipy.stats import norm

import numpy as np


def cost(travel_time):
    """Return a function that computes the cost.

    Args:
        travel_time: Instance of the TravelTime class, which
        determines the travel time function that will be used.

    Returns:
        inner_cost: Function that computes the cost given
        the travel time function.
    """

    def inner_cost(t_a, beta, gamma, t_star):
        """Compute the cost of arriving at a given moment.

        Args:
            t_a: Actual arrival time, in hours
            beta: Early arrival penalty, normalized by the
                value of time.
            gamma: Late arrival penalty, normalized by the
                value of time.
            t_star: Desired arrival time, in hours.

        Returns:
            cost: Cost of the described arrival

        """
        cost = (
            travel_time.f(t_a)
            + beta * jnp.maximum(0, t_star - t_a)
            + gamma * jnp.maximum(0, t_a - t_star)
        )
        return cost

    return inner_cost


def find_td(travel_time):
    """Return a function that computes the optimal arrival time.

    Args:
        travel_time: Instance of the TravelTime class, which
        determines the travel time function that will be used.

    Returns:
        td_from_params: Function that finds the optimal arrival given
        the travel time function.

    """

    def td_from_params(beta, gamma, t_star):
        """Find the optimal arrival time for a single user.

        Args:
            beta: Early arrival penalty, normalized by the
                value of time.
            gamma: Late arrival penalty, normalized by the
                value of time.
            t_star: Desired arrival time, in hours.

        Returns:
            t_a: Optimal arrival time
        """
        cost_fun = cost(travel_time)
        solver = GradientDescent(
            fun=cost_fun, acceleration=False, maxiter=40000, stepsize=0.05
        )
        lval, _ = solver.run(0.0, beta, gamma, t_star)
        rval, _ = solver.run(24.0, beta, gamma, t_star)
        val = jnp.where(
            cost_fun(rval, beta, gamma, t_star)
            < cost_fun(lval, beta, gamma, t_star),
            rval,
            lval,
        )
        t_a = jnp.where(
            cost_fun(val, beta, gamma, t_star)
            < cost_fun(t_star, beta, gamma, t_star),
            val,
            t_star,
        )
        return t_a

    return td_from_params


def generate_arrival(
    n,
    travel_time,
    mu_beta,
    mu_gamma,
    mu_t,
    sigma,
    sigma_t,
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
    # Betas, gammas and t_star are generated according to the chosen
    # distributions

    betas = truncnorm.rvs(
        (0.01 - mu_beta) / sigma,
        10000,
        loc=mu_beta,
        scale=sigma,
        size=n,
        random_state=random_gen,
    )
    gammas = truncnorm.rvs(
        (0.01 - mu_gamma) / sigma,
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
