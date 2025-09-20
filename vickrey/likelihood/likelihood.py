"""Functions concerning the computation of the likelihood."""

from jax import vmap, jit
from jax.nn import relu
from jax.scipy.integrate import trapezoid
from jax.scipy.stats import truncnorm as jtruncnorm
from jax.scipy.stats import norm as jnorm
import jax.numpy as jnp

from vickrey.likelihood.find_points import (
    find_b0,
    find_g0,
    find_be,
    find_gi,
    find_ts,
)


def likelihood(travel_time, t_a, mu_b, mu_g, mu_t, sigma, sigma_t):
    """Find the likelihood of a single data point.

    Given the parameters, find the likelihood of a point realizing a
    minimum. Beta, gamma and t* are assumed to be normally
    distributed.

    Args:
        travel_time: Instance of the TravelTime class the likelihood
            is computed on
        t_a: Time point of which the likelihood is computed
        mu_b: Mean of the beta parameter, relative to the early
            arrival penalty
        mu_g: Mean of the gamma parameter, relative to the late
            arrival penalty
        mu_t: Mean of the t* parameter, relative to the desired
            arrival time
        sigma: Variance of the beta and gamma distributions, that are
            assumed to share the same variance
        sigma_t: Variance of the t* parameter

    Returns:
        likelihood: float number representing the likelihood of an
            optimal arrival being equal to t_a

    """

    # The truncated normals pdf and cdf are defined here
    def cdf_b(b):
        return jtruncnorm.cdf(
            b, (0.01 - mu_b) / sigma, 10000, loc=mu_b, scale=sigma
        )

    def cdf_g(g):
        return jtruncnorm.cdf(
            g, (0.01 - mu_g) / sigma, 10000, loc=mu_g, scale=sigma
        )

    def pdf_b(b):
        return jtruncnorm.pdf(b, (0.01 - mu_b) / sigma, 10000, mu_b, sigma)

    def pdf_g(g):
        return jtruncnorm.pdf(g, (0.01 - mu_g) / sigma, 10000, mu_g, sigma)

    # For computing the probability that a point is a kink minimum, an
    # integral is computed as in the latex.
    b0 = find_b0(t_a, travel_time)
    g0 = find_g0(t_a, travel_time)
    likelihood_kink = (
        jnorm.pdf(t_a, mu_t, sigma_t) * (1 - cdf_b(b0)) * (1 - cdf_g(g0))
    )

    # Now for internal minima: we just follow the equation in the latex.

    # Here, t_a is transformed so that it always returns a plausible
    # result for an early or late arrival. If the actual t_a is not
    # plausible, the pdf of beta will return zero and not yield any
    # problem. These transformations are anyway necessary because
    # impossible values cannot be fed to find_be and find_gi
    # (functions to which this check could be delegated)

    t_a_early = jnp.where(
        jnp.logical_and(travel_time.df(t_a) > 0, travel_time.d2f(t_a) > 0),
        t_a,
        0,
    )
    t_a_late = jnp.where(
        jnp.logical_and(travel_time.df(t_a) < 0, travel_time.d2f(t_a) > 0),
        t_a,
        24,
    )

    def inner_int_early_cdf(x):
        return jnorm.cdf(
            jnp.minimum(
                find_be(t_a_early, travel_time),
                find_ts(travel_time.df(t_a_early), x, travel_time),
            ),
            mu_t,
            sigma_t,
        ) - jnorm.cdf(t_a_early, mu_t, sigma_t)

    def inner_int_early(x):
        return inner_int_early_cdf(x) * pdf_g(x)

    x_gamma = jnp.linspace(1e-2, 3, 500)
    int_early = trapezoid(vmap(inner_int_early)(x_gamma), x_gamma, axis=0)
    lik_early = (
        int_early * pdf_b(travel_time.df(t_a)) * relu(travel_time.d2f(t_a))
    )

    def inner_int_late_cdf(x):
        return jnorm.cdf(t_a_late, mu_t, sigma_t) - jnorm.cdf(
            jnp.maximum(
                find_gi(t_a_late, travel_time),
                find_ts(x, -travel_time.df(t_a_late), travel_time),
            ),
            mu_t,
            sigma_t,
        )

    def inner_int_late(x):
        return inner_int_late_cdf(x) * pdf_b(x)

    x_beta = jnp.linspace(1e-2, 3, 500)

    int_late = trapezoid(vmap(inner_int_late)(x_beta), x_beta, axis=0)

    lik_late = (
        int_late * pdf_g(-travel_time.df(t_a)) * relu(travel_time.d2f(t_a))
    )

    likelihood_internal = lik_early + lik_late
    likelihood = likelihood_kink + likelihood_internal
    return jnp.maximum(likelihood, 1e-31)


def total_liks(travel_time, t_as):
    """Likelihood of each data point in a set of arrival times.

    Args:
        travel_time: Travel time function.
        t_as: Dataset of arrival times.

    Returns:
        mapped_lik: Function mapping a parameter set to the array
            of likelihoods.
    """

    def mapped_lik(mu_b, mu_g, mu_t, sigma, sigma_t):
        @jit
        def lik_restr(t_a):
            return likelihood(
                travel_time, t_a, mu_b, mu_g, mu_t, sigma, sigma_t
            )

        return vmap(lik_restr)(t_as)

    return mapped_lik


def total_log_lik(travel_time, t_as):
    """Total log likelihood of a set of arrival times.

    Args:
        travel_time: Travel time function.
        t_as: Dataset of arrival times.

    Returns:
        mapped_lik: Function mapping a parameter set to the value
            of the total likelihood.
    """

    def mapped_lik(mu_b, mu_g, mu_t, sigma, sigma_t):
        def lik_restr(t_a):
            return likelihood(
                travel_time, t_a, mu_b, mu_g, mu_t, sigma, sigma_t
            )

        return jnp.sum(jnp.log(vmap(lik_restr)(t_as)), axis=0)

    return mapped_lik
