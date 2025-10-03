"""Perform a complete estimation for an approximated travel time function."""

import numpy as np
from time import time

from vickrey.real_data.tt_for_day import RealData
from vickrey.utils import TravelTime
from vickrey.real_data.fit_function import fit_to_data
from vickrey.likelihood.generate_data import generate_arrival
from vickrey.likelihood.optimize import grid_search, grad_free

day = 23
tdata = RealData()
travel_times = tdata.tt_for_day(day)

travel_times.index = travel_times.index / 60
travel_times /= 60
times = np.r_[travel_times.index]
approx = fit_to_data(
    times, travel_times.values, kind="generalized_skewed_gaussian"
)
tt = TravelTime(approx)

par = (0.1, 0.15, 9.5, 0.1, 1.0)

start = time()
t_as = generate_arrival(1000, tt, *par)
print(
    f"In {time() - start:.2f} seconds, generated arrival \
times. Starting grid search..."
)

# %%

start = time()
init = grid_search(t_as, tt)

print(
    f"In {time() - start:.2f} seconds, found initial conditions:\
{init}. Starting iterative optimizer..."
)

start = time()
res = grad_free(t_as, tt, init, True)

print(
    f"\
    In {time() - start:.2f} seconds, optimizer converged to {res.x}. \n\
    Relative errors: {np.abs(np.r_[res.x] - np.r_[par]) / np.r_[par]}\n\
    "
)
