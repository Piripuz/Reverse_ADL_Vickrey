from jax import numpy as jnp
from jax.scipy.stats import gennorm as jgennorm
from jax.lax import logistic
from jax import grad


def gennorm(x, beta, mu, sigma):
    def pdf_unscaled(x, b):
        return jnp.exp(-(jnp.abs(x) ** b))

    y = (x - mu) / sigma
    return pdf_unscaled(y, beta) / sigma


def skewnorm(x, a, mu, sigma):
    def pdf_unscaled(x):
        return 2 * jnp.exp(-(x**2)) * logistic(a * x)

    y = (x - mu) / sigma
    return pdf_unscaled(y) / sigma


def skewgennorm(x, a, beta, mu, sigma):
    def pdf_unscaled(x, beta):
        return 2 * jgennorm.pdf(x, beta) * jgennorm.cdf(a * x, beta)

    y = (x - mu) / sigma
    return pdf_unscaled(y, beta) / sigma


def skewgennorm_fake(x, a, beta, mu, sigma):
    y = (x - mu) / sigma
    gen = gennorm(y, beta, 0, 1)
    log = logistic(a * y)
    return 2 * gen * log / sigma


def _left_hyp(a):
    return lambda x: (jnp.sqrt((x - a[2]) ** 2 + a[0]) + x - a[2]) / a[1]


def _right_hyp(a):
    return lambda x: (jnp.sqrt((x - a[2]) ** 2 + a[0]) - x + a[2]) / a[1]


def _poly_coeffs(a, b, c, p):
    """Compute coefficients for smoothly connecting poly and hyp."""
    k = len(b) + 3
    points = jnp.linspace(*p, k - 1)

    mat = jnp.eye(k + 1)

    mat[0, :] = jnp.r_[0, jnp.arange(1, k + 1) * p[0] ** jnp.arange(k)]
    mat[1, :] = jnp.r_[0, jnp.arange(1, k + 1) * p[1] ** jnp.arange(k)]
    mat[2:, :] = points[:, None] ** jnp.arange(k + 1)

    coeff = jnp.r_[
        grad(_left_hyp(a))(p[0]),
        grad(_right_hyp(c))(p[1]),
        _left_hyp(a)(p[0]),
        b,
        _right_hyp(c)(p[1]),
    ]

    return jnp.linalg.solve(mat, coeff)


def comp_hyp(a, b, c, p, off):
    """Piecewise defined approximating function."""
    bs = _poly_coeffs(a, b, c, p)

    def inner_func(x):
        return (
            jnp.piecewise(
                x,
                [x < p[0], x > p[1]],
                [
                    _left_hyp(a),
                    _right_hyp(c),
                    lambda x: jnp.polyval(jnp.flip(bs), x),
                ],
            )
            + off
        )

    return inner_func
