import os

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

from jax import grad, vmap
from jax import numpy as jnp
from jax.scipy.stats import norm as jnorm

import jax
jax.config.update('jax_enable_x64', True)

def skewnorm_pdf(x, a, mu, sigma):
    pdf_unscaled = lambda x: 2 * jnorm.pdf(x) * jnorm.cdf(a*x)
    y = (x - mu) / sigma
    return pdf_unscaled(y) / sigma

def left_hyp(a):
    return lambda x: (jnp.sqrt((x - a[2])**2 + a[0]) + x - a[2])/a[1]

def right_hyp(a):
    return lambda x: (jnp.sqrt((x - a[2])**2 + a[0]) - x + a[2])/a[1]

def poly_coeffs(a, b, c, p):
    """Given the parameters a, b, c, computes the coefficients of the
    polynomial that smoothly connects with the hyperbolae, and has
    value given by the parameter b at linearly spaced points between
    ps.
    Expects $a, c \in \mathbb{R}^3, b \in \mathbb{R}^{k-3}, p \in
    \mathbb{R}^2$

    """

    k = len(b) + 3
    points = np.linspace(*p, k - 1)
    
    mat = np.eye(k+1)
    mat[0, :] = np.r_[0, np.arange(1, k+1) * p[0] ** np.arange(k)]
    mat[1, :] = np.r_[0, np.arange(1, k+1) * p[1] ** np.arange(k)]
    mat[2:, :] = points[:, None]**np.arange(k+1)

    coeff = np.r_[grad(left_hyp(a))(p[0]),
                  grad(right_hyp(c))(p[1]),
                  left_hyp(a)(p[0]),
                  b,
                  right_hyp(c)(p[1])]

    return np.linalg.solve(mat, coeff)

def func(a, b, c, p, off):
    bs = poly_coeffs(a, b, c, p)
    inner_func = lambda x: \
        jnp.piecewise(x,
                      [x < p[0], x > p[1]],
                      [
                          left_hyp(a),
                          right_hyp(c),
                          # lambda x: (x[:, None]**jnp.arange(len(bs))*bs).sum(axis=1)
                         lambda x: jnp.polyval(jnp.flip(bs), x)
                      ]) + off
    return inner_func

def fit_to_data(x, y, kind="hyperbola", init=None):
    
    supported=["hyperbola", "skewed_gaussian", "generalized_gaussian"]
    if kind not in supported:
        raise ValueError(f"Kind \"{kind}\" is not supported, as it is not among {supported}")

    if kind == "hyperbola":
        to_fit = lambda x, a1, a2, a3, b1, c1, c2, c3, p1, p2, off: \
            func([a1, a2, a3], [b1], [c1, c2, c3], [p1, p2], off)(x)
        if init == None:
            crit_left = x[(y > 1.5*y.min()).argmax()]
            crit_right = x[len(y) - (y > 1.5*y.min())[::-1].argmax()]
            b = y.max()

            p_init = list(np.linspace(crit_left, crit_right, 4)[1:3])
            off_init = y[0]
            a_init = [.01, (p_init[1] - p_init[0])/(b - off_init), crit_left]
            b_init = [b]
            c_init = [*a_init[:2], crit_right]
        else:
            a_init = init[:3]
            b_init = init[3] 
            c_init = init[4:7]
            p_init = init[7:9]
            off_init = init[9]

        popt, _ = curve_fit(to_fit,
                            x,
                            y,
                            a_init + b_init + c_init + p_init + [off_init],
                            bounds=(
                                [0, 0, -np.inf] + [-np.inf]*len(b_init) + [0, 0, -np.inf, 0, 0, 0],
                                [np.inf]*(9+len(b_init))
                            ))
        a, b, c, p, off = popt[:3], popt[3:(3 + len(b_init))], popt[-6:-3], popt[-3:-1], popt[-1]
        print(
            f"\
            Converged to:\n\
            a = {a}\n\
            b = {b}\n\
            c = {c}\n\
            p = {p}\n\
            offset = {off}\
            "
        )
        return func(a, b, c, p, off)
    elif kind == "skewed_gaussian":
        if init == None:
            crit_left = x[(y > 1.5*y.min()).argmax()]
            crit_right = x[len(y) - (y > 1.5*y.min())[::-1].argmax()]

            a_init = 0
            mu_init = x[y.argmax()]
            sigma_init = (crit_right - crit_left)/4
            off_init = y[0]
            scale_init =  (y.max() - off_init) / skewnorm_pdf(mu_init, a_init, mu_init, sigma_init)
        else:
            a_init, mu_init, sigma_init, scale_init, off_init = init

        to_fit = lambda x, a, mu, sigma, scale, off: skewnorm_pdf(x, a, mu, sigma)*scale + off
        popt, _ = curve_fit(
            to_fit,
            x,
            y,
            [a_init, mu_init, sigma_init, scale_init, off_init],
        )
        a, mu, sigma, scale, off = popt
        print(
            f"\
            Converged to:\n\
            a = {a}\n\
            mu = {mu}\n\
            sigma = {sigma}\n\
            scale = {scale}\n\
            offset = {off}\
            "
        )
        return lambda x: to_fit(x, a, mu, sigma, scale, off)
    elif kind == "generalized_gaussian":
        raise NotImplementedError()
