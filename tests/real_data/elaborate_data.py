import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = "cycles_result_vars_low_2"

res = pd.read_csv(os.path.join(base_dir, "results.csv"), header=None)
par = pd.read_csv(os.path.join(base_dir, "parameters.csv"), header=None)
days = pd.read_csv(os.path.join(base_dir, "days.csv"), header=None)[0]

# %%
err = res - par
# err = err.drop(1)
err_rel = err / par.mean(axis=0)
err_arr = err.mean(axis=1)
# %%
max = 0
for i in range(5):
    for j in range(5):
        curr = np.corrcoef(err_rel[i], par[j])[0, 1]
        if np.abs(curr) > max:
            max = curr
            max_index = (i, j)
# %%
fig, ax = plt.subplots(figsize=(4, 6))
labels = r"$\mu_\gamma$ $\mu_\beta$ $\mu_t$ $\sigma$ $\sigma_t$".split(" ")
ax.boxplot(
    err_rel,
    tick_labels=labels,
    patch_artist=True,
    flierprops={"marker": "x", "markeredgecolor": "darkred", "markersize": 5},
)
ax.axhline(0, color="black", linewidth=0.2)
ax.set_ylim((-0.4, 0.6))

# fig.savefig('boxplot.png', dpi=400)

plt.show()
