import numpy as np
import matplotlib.pyplot as plt

from vickrey.real_data.tt_for_day import RealData
from vickrey.utils import TravelTime
from vickrey.fit_function import fit_to_data
from vickrey.likelihood.generate_data import generate_arrival
from vickrey.likelihood.optimize import grid_search, grad_free
from vickrey.likelihood.plot import plot_contour
day = 23
tdata = RealData()
travel_times = tdata.tt_for_day(day)

travel_times.index = travel_times.index/60
travel_times /= 60
times = np.r_[travel_times.index]
approx = fit_to_data(times, travel_times.values)
tt = TravelTime(approx)

par = [.05, .3, 9.5, .1, 1.]

_, _, _, t_as = generate_arrival(1000, tt, *par)

print("Generated arrival times. Starting grid search...")

init = grid_search(t_as, tt)

print(f"Initial conditions found: {init}. Starting iterative optimizer...")

res = grad_free(t_as, tt, init)

print(f"Optimizer converged to {res.x}")


fig, ax = plt.subplots(figsize=(6, 4))
plot_contour(tt, t_as, ax, x_bounds=[.01, .3], y_bounds=[.01, .5])
ax.scatter(*res.x[:2])
fig.show()
