import numpy as np
from scipy.optimize import curve_fit


from vickrey.functions import gennorm, skewnorm, skewgennorm, comp_hyp

# import jax
# jax.config.update("jax_enable_x64", True)


def fit_to_data(x, y, kind="hyperbola", init=None):
    supported = [
        "hyperbola",
        "skewed_gaussian",
        "generalized_gaussian",
        "generalized_skewed_gaussian",
        "skewed_generalized_gaussian",
    ]
    if kind not in supported:
        raise ValueError(
            f'Kind "{kind}" is not supported, as it is not among {supported}'
        )

    if kind == "hyperbola":

        def to_fit(x, a1, a2, a3, b1, c1, c2, c3, p1, p2, off):
            return comp_hyp([a1, a2, a3], [b1], [c1, c2, c3], [p1, p2], off)(x)

        if init is None:
            crit_left = x[(y > 1.5 * y.min()).argmax()]
            crit_right = x[len(y) - (y > 1.5 * y.min())[::-1].argmax()]
            b = y.max()

            p_init = list(np.linspace(crit_left, crit_right, 4)[1:3])
            off_init = y[0]
            a_init = [
                0.01,
                (p_init[1] - p_init[0]) / (b - off_init),
                crit_left,
            ]
            b_init = [b]
            c_init = [*a_init[:2], crit_right]
        else:
            a_init = init[:3]
            b_init = init[3]
            c_init = init[4:7]
            p_init = init[7:9]
            off_init = init[9]

        popt, _ = curve_fit(
            to_fit,
            x,
            y,
            a_init + b_init + c_init + p_init + [off_init],
            bounds=(
                [0, 0, -np.inf]
                + [-np.inf] * len(b_init)
                + [0, 0, -np.inf, 0, 0, 0],
                [np.inf] * (9 + len(b_init)),
            ),
        )
        a, b, c, p, off = (
            popt[:3],
            popt[3 : (3 + len(b_init))],
            popt[-6:-3],
            popt[-3:-1],
            popt[-1],
        )
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
        return comp_hyp(a, b, c, p, off)
    elif kind == "skewed_gaussian":
        if init is None:
            crit_left = x[(y > 1.5 * y.min()).argmax()]
            crit_right = x[len(y) - (y > 1.5 * y.min())[::-1].argmax()]

            a_init = 0
            mu_init = x[y.argmax()]
            sigma_init = (crit_right - crit_left) / 4
            off_init = y[0]
            scale_init = (y.max() - off_init) / skewnorm(
                mu_init, a_init, mu_init, sigma_init
            )
        else:
            if len(init) != 5:
                raise ValueError(
                    f"Initial conditions for function\
                fitting are long {len(init)}, length 5 was expected."
                )
            a_init, mu_init, sigma_init, scale_init, off_init = init

        def to_fit(x, a, mu, sigma, scale, off):
            return skewnorm(x, a, mu, sigma) * scale + off

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
            offset = {off}\n\
            "
        )
        return lambda x: to_fit(x, a, mu, sigma, scale, off)
    elif kind == "generalized_gaussian":
        if init is None:
            crit_left = x[(y > 1.5 * y.min()).argmax()]
            crit_right = x[len(y) - (y > 1.5 * y.min())[::-1].argmax()]

            beta_init = 2
            mu_init = x[y.argmax()]
            sigma_init = (crit_right - crit_left) / 4
            off_init = y[0]
            scale_init = (y.max() - off_init) / gennorm(
                mu_init, beta_init, mu_init, sigma_init
            )
        else:
            if len(init) != 5:
                raise ValueError(
                    f"Initial conditions for function\
                fitting are long {len(init)}, length 5 was expected."
                )
            beta_init, mu_init, sigma_init, scale_init, off_init = init

        def to_fit(x, beta, mu, sigma, scale, off):
            return gennorm(x, beta, mu, sigma) * scale + off

        popt, _ = curve_fit(
            to_fit,
            x,
            y,
            [beta_init, mu_init, sigma_init, scale_init, off_init],
        )
        beta, mu, sigma, scale, off = popt
        print(
            f"\
            Converged to:\n\
            beta = {beta}\n\
            mu = {mu}\n\
            sigma = {sigma}\n\
            scale = {scale}\n\
            offset = {off}\n\
            "
        )
        return lambda x: to_fit(x, beta, mu, sigma, scale, off)

    elif kind in (
        "skewed_generalized_gaussian",
        "generalized_skewed_gaussian",
    ):
        if init is None:
            crit_left = x[(y > 1.5 * y.min()).argmax()]
            crit_right = x[len(y) - (y > 1.5 * y.min())[::-1].argmax()]

            beta_init = 2
            a_init = 0
            mu_init = x[y.argmax()]
            sigma_init = (crit_right - crit_left) / 4
            off_init = y[0]
            scale_init = (y.max() - off_init) / gennorm(
                mu_init, beta_init, mu_init, sigma_init
            )
        else:
            if len(init) != 6:
                raise ValueError(
                    f"Initial conditions for function\
                fitting are long {len(init)}, length 6 was expected."
                )
            beta_init, a_init, mu_init, sigma_init, scale_init, off_init = init

        def to_fit(x, a, beta, mu, sigma, scale, off):
            return skewgennorm(x, a, beta, mu, sigma) * scale + off

        popt, _ = curve_fit(
            to_fit,
            x,
            y,
            [a_init, beta_init, mu_init, sigma_init, scale_init, off_init],
        )
        a, beta, mu, sigma, scale, off = popt
        print(
            f"\
            Converged to:\n\
            a = {a}\n\
            beta = {beta}\n\
            mu = {mu}\n\
            sigma = {sigma}\n\
            scale = {scale}\n\
            offset = {off}\n\
            "
        )
        return lambda x: to_fit(x, a, beta, mu, sigma, scale, off)
