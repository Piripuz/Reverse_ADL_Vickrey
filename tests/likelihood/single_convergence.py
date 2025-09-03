from vickrey.likelihood.optimize import grid_search, grad_free
from vickrey.likelihood.generate_data import generate_arrival
from vickrey.travel_times import asymm_gaussian
from vickrey.utils import TravelTime

num = 1000
tt = TravelTime(asymm_gaussian())
par = (0.8, 0.4, 9.0, 0.4, 1.5)

t_as = generate_arrival(num, tt, *par)

print("Generated arrivals")

init = grid_search(t_as, tt)

print(
    f"\
    Finished grid search.\n\
    Initial contition found: \
    {init}\
    "
)

res = grad_free(t_as, tt, init)
print(f"Converged to {res.x} in {res.nit} iterations")
