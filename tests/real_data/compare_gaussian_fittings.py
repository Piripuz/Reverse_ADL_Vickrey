import argparse
import matplotlib.pyplot as plt
import numpy as np

from vickrey.real_data.fit_function import fit_to_data
from vickrey.real_data.tt_for_day import RealData
from vickrey.utils import is_weekend

parser = argparse.ArgumentParser("compare_gaussian_fittings")
parser.add_argument(
    "day", help="Day for which the travel time will be plotted", type=int
)
args = parser.parse_args()


def plot_fit(day, save=False):
    data = RealData().tt_for_day(day)
    data.index = data.index / 60
    data /= 60
    times = np.r_[data.index]

    skewed = fit_to_data(times, data.values, kind="skewed_gaussian")
    generalized = fit_to_data(times, data.values, kind="generalized_gaussian")
    skewed_generalized = fit_to_data(
        times, data.values, kind="skewed_generalized_gaussian"
    )

    y_sk = skewed(times)
    y_gen = generalized(times)
    y_skgen = skewed_generalized(times)

    err_sk = np.sum((y_sk - data.values) ** 2) / len(data)
    err_gen = np.sum((y_gen - data.values) ** 2) / len(data)
    err_skgen = np.sum((y_skgen - data.values) ** 2) / len(data)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data, label="Real data")
    ax.plot(
        times, y_sk, label=f"Skewed Gaussian (MSE = {err_sk:.2e})", alpha=0.5
    )
    ax.plot(
        times,
        y_gen,
        label=f"Generalized Gaussian (MSE = {err_gen:.2e})",
        alpha=0.5,
    )
    ax.plot(
        times,
        y_skgen,
        label=f"Skewed generalized Gaussian (MSE = {err_skgen:.2e})",
    )
    ax.set_xlabel("Departure time (h)")
    ax.set_ylabel("Travel time (h)")
    ax.set_title(data.name)
    ax.legend(loc=1)
    if save:
        fig.savefig(
            f"/home/piripuz/tmp/interpolations/interpolations_day_{day}.png",
            dpi=300,
        )
    else:
        fig.show()


day = args.day

if day is None:
    while True:
        day = np.random.randint(1, 181)
        if is_weekend(day):
            continue
        plot_fit(day, save=False)
        plt.show()

plot_fit(day)
plt.show()
