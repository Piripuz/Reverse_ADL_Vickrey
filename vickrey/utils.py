from jax import grad, vmap
import jax.numpy as jnp

from jaxopt import GradientDescent

class TravelTime():
    def __init__(self, function, df=None, d2f=None):
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
    def inner_step(iter_num):
        return jnp.where(iter_num < nhigh, high,
                         jnp.where(iter_num < nsmall+nhigh, small, vsmall))
    return inner_step
                         
