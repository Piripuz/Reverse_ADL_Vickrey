"""Various utilities for the vickrey package."""

from jax import grad, vmap
import jax.numpy as jnp

from jaxopt import GradientDescent

from datetime import datetime


class TravelTime:
    """Structure for a travel time function."""

    def __init__(self, function, df=None, d2f=None):
        """Save the travel time function, and computes relevant coefficients.

        Args:
            function: Travel time provile function.
            df (optional): First derivative of the function
            d2f (optional): second derivative of the function
        """
        self.f = function
        if df is None:
            self.df = grad(function)
        else:
            self.df = df

        if d2f is None:
            self.d2f = grad(grad(function))
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
    """Return a varying step size to be used in iterative optimizers."""

    def inner_step(iter_num):
        return jnp.where(
            iter_num < nhigh,
            high,
            jnp.where(iter_num < nsmall + nhigh, small, vsmall),
        )

    return inner_step


def is_weekend(day):
    """Compute wether a day of 2017 is a weekend day."""
    date = datetime.strptime(f"{day:03}" + "2017", "%j%Y")
    return date.weekday() > 4
