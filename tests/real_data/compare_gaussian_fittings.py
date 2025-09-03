import matplotlib.pyplot as plt
import numpy as np
from vickrey.fit_function import fit_to_data
from vickrey.real_data.tt_for_day import RealData


def plot_fit(day):
    data = RealData().tt_for_day(day)
    data.index = data.index / 60
    data /= 60
    times = np.r_[data.index]

    skewed = fit_to_data(times, data.values, kind="skewed_gaussian")
    generalized = fit_to_data(times, data.values, kind="generalized_gaussian")

    y_sk = skewed(times)
    y_gen = generalized(times)

    err_sk = np.sum((y_sk - data.values) ** 2) / len(data)
    err_gen = np.sum((y_gen - data.values) ** 2) / len(data)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data, label="Real data")
    ax.plot(times, y_sk, label=f"Skewed Gaussian (MSE = {err_sk:.2e})")
    ax.plot(times, y_gen, label=f"Generalized Gaussian (MSE = {err_gen:.2e})")
    ax.set_xlabel("Departure time (h)")
    ax.set_ylabel("Travel time (h)")
    ax.set_title(data.name)
    ax.legend(loc=1)
    fig.savefig(
        f"/home/piripuz/tmp/interpolations/interpolations_day_{day}.png",
        dpi=300,
    )


for day in range(23, 28):
    plot_fit(day)
