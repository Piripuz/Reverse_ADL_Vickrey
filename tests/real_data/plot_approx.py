import numpy as np
import matplotlib.pyplot as plt

from vickrey.real_data.fit_function import fit_to_data
from vickrey.real_data.tt_for_day import RealData

travel_data = RealData()
kinds = [
    "skewed_gaussian",
    "generalized_gaussian",
    "skewed_generalized_gaussian",
]


def plot_fittings(day, ax):
    travel_times = travel_data.tt_for_day(day) / 60
    travel_times.index = travel_times.index / 60
    times = np.r_[travel_times.index]
    ax.plot(times, travel_times.values)
    for i, kind in enumerate(kinds):
        approx = fit_to_data(times, travel_times.values, kind=kind)
        y = approx(times)
        ax.plot(times, y, label=" ".join(kind.split("_")).capitalize())


fig, ax = plt.subplots(figsize=(6, 4))
plot_fittings(24, ax)
ax.legend()
fig.show()
