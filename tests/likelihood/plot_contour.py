import matplotlib.pyplot as plt

from vickrey.likelihood.generate_data import generate_arrival
from vickrey.likelihood.plot import plot_contour
from vickrey.utils import TravelTime
from vickrey.travel_times import asymm_gaussian

tt = TravelTime(asymm_gaussian())
_, _, _, t_as = generate_arrival(1000, tt)
print("Generated arrival times. Computing values for plotting...")
ax = plot_contour(
    tt,
    t_as,
)
plt.show()
