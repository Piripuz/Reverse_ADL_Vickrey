from jax import grad, vmap
from jax.nn import relu
from jax.scipy.integrate import trapezoid
from jax.scipy.stats import truncnorm as jtruncnorm
from jax.scipy.stats import norm as jnorm
import jax.numpy as jnp

from vickrey.likelihood.find_points import find_bs, find_gs, find_b0, find_g0, find_be, find_gi, find_ts

def likelihood(travel_time, t_a, mu_b, mu_g, mu_t, sigma, sigma_t):
    """Finds the likelihood of a point realizing a minimum, for the
    travel time, beta, gamma and t* distributions determined by the
    parameters.  Beta, gamma and t* are assumed to be normally
    distributed.

    """

    # The truncated normals pdf and cdf are definde here
    # cdf_b = lambda b: jnorm.cdf(b, mu_b, sigma)
    # cdf_g = lambda g: jnorm.cdf(g, mu_g, sigma)
    # pdf_b = lambda b: jnorm.pdf(b, mu_b, sigma)
    # pdf_g = lambda g: jnorm.pdf(g, mu_g, sigma)
    cdf_b = lambda b: jtruncnorm.cdf(b, -mu_b / sigma, 10000, loc=mu_b, scale=sigma)
    cdf_g = lambda g: jtruncnorm.cdf(g, -mu_g / sigma, 10000, loc=mu_g, scale=sigma)
    pdf_b = lambda b: jtruncnorm.pdf(b, -mu_b / sigma, 10000, mu_b, sigma)
    pdf_g = lambda g: jtruncnorm.pdf(g, -mu_g / sigma, 10000, mu_g, sigma)

    # For computing the probability that a point is a kink minimum, an
    # integral is computed as in the latex.
    b0 = find_b0(t_a, travel_time)
    g0 = find_g0(t_a, travel_time)
    likelihood_kink = jnorm.pdf(t_a, mu_t, sigma_t) * (1 - cdf_b(b0)) * (1 - cdf_g(g0))

    # Now for internal minima: we just follow the equation in the latex.

    # Here, t_a is transformed so that it always returns a plausible
    # result for an early or late arrival. If the actual t_a is not
    # plausible, the pdf of beta will return zero and not yield any
    # problem. These transformations are anyway necessary because
    # impossible values cannot be fed to find_be and find_gi
    # (functions to which this check could be delegated)
    
    t_a_early = jnp.where(jnp.logical_and(travel_time.df(t_a) > 0, travel_time.d2f(t_a) > 0), t_a, 0)
    t_a_late = jnp.where(jnp.logical_and(travel_time.df(t_a) < 0, travel_time.d2f(t_a) > 0), t_a, 24)
    
    inner_int_early_cdf = lambda x:  jnorm.cdf(jnp.minimum(find_be(t_a_early, travel_time), find_ts(travel_time.df(t_a_early), x, travel_time)), mu_t, sigma_t) - jnorm.cdf(t_a_early, mu_t, sigma_t)

    inner_int_early = lambda x: inner_int_early_cdf(x) * pdf_g(x)

    x_gamma = jnp.linspace(1e-2, 10, 200)

    int_early = trapezoid(vmap(inner_int_early)(x_gamma), x_gamma, axis=0)
    lik_early = int_early * pdf_b(travel_time.df(t_a)) * relu(travel_time.d2f(t_a))
    inner_int_late_cdf = lambda x: jnorm.cdf(t_a_late, mu_t, sigma_t) - jnorm.cdf(jnp.maximum(find_gi(t_a_late, travel_time), find_ts(x, -travel_time.df(t_a_late), travel_time)), mu_t, sigma_t)

    inner_int_late = lambda x: inner_int_late_cdf(x) * pdf_b(x)

    x_beta = jnp.linspace(1e-2, 10, 200)

    int_late = trapezoid(vmap(inner_int_late)(x_beta), x_beta, axis=0)

    lik_late = int_late * pdf_g(-travel_time.df(t_a)) * relu(travel_time.d2f(t_a))

    likelihood_internal = lik_early + lik_late
    likelihood = likelihood_kink + likelihood_internal
    return jnp.maximum(likelihood, 1e-31)

def total_liks(travel_time, t_as):
    def mapped_lik(mu_b, mu_g, mu_t, sigma, sigma_t):
        lik_restr = lambda t_a: likelihood(travel_time, t_a, mu_b, mu_g, mu_t, sigma, sigma_t)
        return vmap(lik_restr)(t_as)
    return mapped_lik

def total_log_lik(travel_time, t_as):
    def mapped_lik(mu_b, mu_g, mu_t, sigma, sigma_t):
        lik_restr = lambda t_a: likelihood(travel_time, t_a, mu_b, mu_g, mu_t, sigma, sigma_t)
        return jnp.sum(jnp.log(vmap(lik_restr)(t_as)), axis=0)
    return mapped_lik
