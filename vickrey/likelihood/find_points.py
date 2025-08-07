from jaxopt import GradientDescent, Bisection
import jax.numpy as jnp
from jax.lax import while_loop

from vickrey.utils import steps

def find_bs(beta, travel_time):
    """ Given a travel time function and a beta value,
    finds the interval in which the optimal arrival time is constant
    (and there are thus no kink minima).

    Returns a couple containing initial and final points of the interval
    """
    
    # A gradient descent algorithm finds the initial point. The step
    # size is chosen to be inversely proportional to the value of beta
    # because the function will become shallower the lower the beta
    # is.
    stepsize = lambda n: steps()(n)/beta
    in_obj = lambda x: travel_time.f(x) - beta*x
    solver = GradientDescent(fun=in_obj, acceleration=False, stepsize=stepsize)
    b_i, _ = solver.run(0.)
    b_e = find_be(b_i, travel_time)
    # The interval extremes are returned
    return jnp.r_[b_i, b_e]

def find_be(b_i, travel_time):
    # The final point is found where the line starting from the initial point,
    # whith slope beta, crosses the travel time function.
    # This point is found via a bisection

    beta = travel_time.df(b_i)

    fin_obj = lambda x: travel_time.f(x) - beta*(x - b_i) - travel_time.f(b_i)

    # Two points where to start the bisection are computed
    step = .5
    high = while_loop(lambda a: fin_obj(a) > 0, lambda a: a + step, b_i + step)
    low = high - step
    return Bisection(fin_obj, low, high, check_bracket=False).run().params
    
    
def find_b0(t_a, travel_time):
    """ Given an arrival time, finds the maximal beta such that
    the arrival time is a kink equilibrium for every value higher than the one returned.
    
    Finds the parameter by bisection.
    """

    # A really low and a really high value are defined as starting points for the bisection
    min = 1e-2
    max = travel_time.maxb

    # The objective function, an indicator function that shows wether the parameter t_a
    # is in the interval for a given beta, is defined
    isin = lambda x, int: jnp.where(jnp.logical_and((x > int[0]), (x < int[1])), 1, -1)
    isin_obj = lambda b: isin(t_a, find_bs(b, travel_time))

    # If t_a is not in the interval for the starting points,
    # the starting points are returned themselves.
    # Otherwise, the bisection algorithm is run.
    is_max = isin_obj(min) == -1
    is_min = isin_obj(max) == 1
    sol = jnp.where(
        jnp.logical_and(
            jnp.logical_not(is_max),
            jnp.logical_not(is_min)),
        Bisection(isin_obj, min, max, check_bracket=False).run().params,
        jnp.where(is_max, min, max))
    return sol


def find_gs(gamma, travel_time):
    """Given a travel time function and a gamma value,
    finds the interval in which the optimal arrival time is constant
    (and there are thus no kink minima).

    Returns a couple containing initial and final points of the interval.

    NOTE: This function's result will be meaningless if called on
    values of gamma greater than the maximum steep of the negative
    travel time function. Don't do that

    """
    
    # A gradient descent algorithm finds the final point
    stepsize = lambda n: steps()(n)/gamma
    fin_obj = lambda x: travel_time.f(x) + gamma*x
    solver = GradientDescent(fun=fin_obj, acceleration=False, stepsize=stepsize)
    g_e, state = solver.run(24.)
    g_i = find_gi(g_e, travel_time)
    # The interval extremes are returned
    return jnp.r_[g_i, g_e]

def find_gi(g_e, travel_time):
    # The initial point is found where the line starting from the
    # final point, whith slope -gamma, crosses the travel time
    # function.
    # This point is found via a bisection

    gamma = -travel_time.df(g_e)

    fin_obj = lambda x: travel_time.f(x) + gamma*(x - g_e) - travel_time.f(g_e)

    # Two points where to start the bisection are computed
    step = .5
    low = while_loop(lambda a: fin_obj(a) > 0, lambda a: a - step, g_e - step)
    high = low + step
    return Bisection(fin_obj, low, high, check_bracket=False).run().params



def find_g0(t_a, travel_time):
    """Given an arrival time, finds the maximal gamma such that the
    arrival time is a kink equilibrium for every value lower than the
    one returned.
    
    Finds the parameter by bisection.

    """

    # A really low and a really high value are defined as starting
    # points for the bisection
    min = 1e-2
    max = travel_time.maxg

    # The objective function, an indicator function that shows wether
    # the parameter t_a is in the interval for a given gamma, is
    # defined
    isin = lambda x, int: jnp.where(jnp.logical_and((x > int[0]), (x < int[1])), 1, -1)
    isin_obj = lambda g: isin(t_a, find_gs(g, travel_time))

    # If t_a is not in the interval for the starting points,
    # the starting points are returned themselves.
    # Otherwise, the bisection algorithm is run.
    is_max = isin_obj(min) == -1
    is_min = isin_obj(max) == 1
    sol = jnp.where(
        jnp.logical_and(
            jnp.logical_not(is_max),
            jnp.logical_not(is_min)),
        Bisection(isin_obj, min, max, check_bracket=False).run().params,
        jnp.where(is_max, min, max))
    return sol

def find_ts(beta, gamma, travel_time):
    gamma += 1e-15
    beta += 1e-15
    # Here, calls to find_bs and find_gs with invalid values have to
    # be avoided: if gamma is too big, ts should return infinite (or
    # 24)
    gamma_adapted = jnp.minimum(gamma, travel_time.maxg - 1e-2)
    beta_adapted = jnp.minimum(beta, travel_time.maxb - 1e-2)
    b_i, b_e = find_bs(beta_adapted, travel_time)
    g_i, g_e = find_gs(gamma_adapted, travel_time)
    num = beta * b_i - travel_time.f(b_i) + gamma * g_e + travel_time.f(g_e)
    res = num / (beta + gamma)
    return jnp.where(gamma < travel_time.maxg, jnp.where(beta < travel_time.maxb, res, 0), 24)
