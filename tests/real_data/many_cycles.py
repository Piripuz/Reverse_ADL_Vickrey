"""Perform several optimization cycles."""

import numpy as np
from time import time
import os

from vickrey.utils import is_weekend
from vickrey.real_data.fit_function import tt_data
from vickrey.likelihood.generate_data import generate_arrival
from vickrey.likelihood.optimize import optim_cycle

base_dir = "cycles_result_vars_low_2"
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

file_res = os.path.join(base_dir, "results.csv")
file_pars = os.path.join(base_dir, "parameters.csv")
file_days = os.path.join(base_dir, "days.csv")

if (
    os.path.isfile(file_res)
    or os.path.isfile(file_pars)
    or os.path.isfile(file_days)
):
    choice = "n"
    while choice:
        choice = input(
            "One of the files already exists. Continue appending? [Y/n]\n"
        )
        if choice == "n":
            raise OSError("One of the files already exists.")
        if choice.lower() == "y":
            continue

num_it = 70
num_agents = 1000

errs = np.zeros((num_it, 5))
days = []
while len(days) < num_it:
    day = np.random.randint(1, 181)
    if not is_weekend(day):
        days.append(day)

for i, day in enumerate(days):
    start = time()
    print(f"Estimating day {day}...\n")
    tt = tt_data(day)
    # Akward structure in if for dealing with case in which
    # maxg, maxb == np.nan
    if (not tt.maxg > 0.08) or (not tt.maxb > 0.08):
        print(f"Discarding day {day}, as g = {tt.maxg}, b = {tt.maxb}\n")
        continue
    par = np.random.uniform(
        [0.05, 0.05, 7.5, 0.01, 0.08],
        [tt.maxb, tt.maxg, 10, np.minimum(tt.maxb, tt.maxg) / 3, 1],
    )
    t_as = generate_arrival(num_agents, tt, *par)
    res = optim_cycle(t_as, tt, par)
    minutes = (time() - start) / 60
    print(
        (f"Iteration {i} completed in {minutes:.2f} minutes. Saving data...\n")
    )
    with open(file_res, "ab") as f:
        np.savetxt(f, res.x[None], fmt="%.6f", delimiter=",")
    with open(file_pars, "ab") as f:
        np.savetxt(f, par[None], fmt="%.6f", delimiter=",")
    with open(file_days, "a") as f:
        f.write(f"{day}\n")
