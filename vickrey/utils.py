from jax import grad, vmap
import jax.numpy as jnp

from jaxopt import GradientDescent


def fuzzy_memoize(fun, start=0, stop=24, prec=2):
    """Provide fuzzy memoization for a real function.

    Args:
        fun: A real function of real numbers
        start, stop (default=0, 24): The bounds of the function domain
        prec (default=2): The density of the grid of the lookup table

    Returns:
        memoized_fun: An approximation of the function `fun` which,
        instead of computing the result at each function call,
        approximates the results by using a previously computed
        lookup table

    """
    grid = jnp.arange(start, stop, 10**-prec)
    res = vmap(fun)(grid)

    def memoized_fun(x):
        rounded_arg = jnp.round(x, prec)
        index = jnp.astype(rounded_arg * 100, jnp.int16) - start
        return res[index]

    return memoized_fun


class TravelTime:
    def __init__(self, function, df=None, d2f=None):
        self.f = fuzzy_memoize(function)
        if df is None:
            self.df = fuzzy_memoize(grad(function))
        else:
            self.df = df

        if d2f is None:
            self.d2f = fuzzy_memoize(grad(grad(function)))
        else:
            self.d2f = d2f

        self.maxb, self.maxg = self.__find_ders()

    def __find_ders(self):
        x = jnp.linspace(0, 24, 100)
        init_b = x[jnp.argmax(vmap(self.df)(x))]
        max, _ = GradientDescent(lambda x: -self.df(x)).run(init_b)
        init_g = x[jnp.argmin(vmap(self.df)(x))]
        min, _ = GradientDescent(self.df).run(init_g)
        return self.df(max), -self.df(min)


def steps(high=0.1, nhigh=200, small=0.01, nsmall=450, vsmall=1e-3):
    def inner_step(iter_num):
        return jnp.where(
            iter_num < nhigh,
            high,
            jnp.where(iter_num < nsmall + nhigh, small, vsmall),
        )

    return inner_step
