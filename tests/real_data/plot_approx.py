import numpy as np
import matplotlib.pyplot as plt

from vickrey.fit_function import fit_to_data
from vickrey.real_data.tt_for_day import RealData

travel_data = RealData()

# %%

day = 33
travel_times = travel_data.tt_for_day(day)

times = np.r_[travel_times.index]

approx = fit_to_data(times, travel_times.values)


plt.plot(travel_times)
plt.plot(times, approx(times))
plt.show()
