"""Plots an histogram comparing empirical and theoretical likelihood."""

import matplotlib.pyplot as plt
import numpy as np
from time import time

from vickrey.travel_times import asymm_gaussian_plateau
from vickrey.utils import TravelTime
from vickrey.likelihood.generate_data import generate_arrival
from vickrey.likelihood.likelihood import total_liks

n = 10000
par = 0.7, 1.2, 9.5, 0.1, 1.0

tt = TravelTime(asymm_gaussian_plateau())
tt.f(7.5)
data = generate_arrival(n, tt, *par)
x = np.linspace(data.min(), data.max(), 200)

now = time()
liks = total_liks(tt, x)(*par)

plt.hist(data, 300)
plt.fill_between(x, liks * 250, color="red", alpha=0.3)
plt.show()
