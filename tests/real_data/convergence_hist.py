from jax import jit
from scipy.optimize import minimize
import numpy as np
import os
import matplotlib.pyplot as plt

from vickrey.likelihood.generate_data import generate_arrival
from vickrey.likelihood.optimize import grid_search
from vickrey.real_data.fit_function import tt_data
from vickrey.likelihood.plot import plot_hist
from vickrey.likelihood.likelihood import total_log_lik

par = [0.113539, 0.170671, 9.865652, 0.018393, 0.851349]
n = 1000
day = 23

tt = tt_data(day)
t_as = generate_arrival(n, tt, *par)


fev = []


@jit
def lik_fun(x):
    return -total_log_lik(tt, t_as)(*x)


def target_fun(x):
    print(x)
    fev.append(x)
    return lik_fun(x)


init = grid_search(t_as, tt, value_sigma="high", value_sigma_t="high")
res = minimize(target_fun, init, method="Nelder-Mead")
fev_high = np.r_[fev]

fev = []

init = grid_search(t_as, tt, value_sigma="low", value_sigma_t="low")
res = minimize(target_fun, init, method="Nelder-Mead")
fev_low = np.r_[fev]
# %%
base_dir = "hist_animations"
high_dir = os.path.join(base_dir, "high_var")
low_dir = os.path.join(base_dir, "low_var")

for dir in [base_dir, high_dir, low_dir]:
    if not os.path.exists(dir):
        os.mkdir(dir)


def save_hists(fev, path):
    n = len(fev)
    for i, par in enumerate(fev):
        filename = os.path.join(path, f"hist_{i:03}.png")
        plot_hist(tt, t_as, par, factor=80)
        plt.savefig(filename, dpi=400)
        plt.close("all")
        print(f"Saved histogram {i + 1}/{n}\n")


save_hists(fev_high, high_dir)
save_hists(fev_low, low_dir)
