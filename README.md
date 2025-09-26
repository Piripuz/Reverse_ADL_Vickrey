# Reverse ADL-Vickrey

This repository hosts all the code used in the paper ().
The code allows estimation of the users' scheduling delays preferences, 
as well as desired arrival time and their variances, from data about arrival time.
The estimation is done by performing a Maximum Likelihood Estimation.

Documentation for the package is available at <https://piripuz.github.io/Reverse_ADL_Vickrey/vickrey/>.
The source for the documentation is available [here](docs/vickrey).

## Installation

The software can be installed as a normal Python package, via `pip`:
simply clone the repository, and run
```
pip install .
```
in the root directory.

The only module the package provides is the `vickrey` module.

## Usage

The package provides functions for performing estimation in three different steps:
- [Definition of the Travel Time Function](#travel-time-function)
- [Generation of synthetic data](#generating-data)
- [Optimization of the Likelihood](#likelihood-optimization)

Scripts in the [tests](tests/) folder provide example usage for how the tasks can be performed together.

### Travel Time Function

For being used, the travel time functions have to be _encapsulated_ in a [`TravelTime` class](https://piripuz.github.io/Reverse_ADL_Vickrey/vickrey/utils.html#vickrey.utils.TravelTime),
which eases the access to the relevant parameters.
This can be done in two different ways, depending on the type of function that is desired.

- For purely __theoretical travel time functions__,
	the constructor of the `TravelTime` class can be directly used.
	Some relevant travel time profile are defined in the module [`vickrey.travel_times`](https://piripuz.github.io/Reverse_ADL_Vickrey/vickrey/travel_times.html).

- For __realistic travel time functions__,
	the function [`vickrey.real_data.fit_function.tt_data`](https://piripuz.github.io/Reverse_ADL_Vickrey/vickrey/real_data/fit_function.html#vickrey.real_data.fit_function.tt_data) can be used.
	It automatically fit the function of a wanted kind to the wanted data,
	and returns the `TravelTime` object containing the fitted function.

### Generating Data

To generate the arrival times data, the used function is [`vickrey.likelihood.generate_data.generate_arrival`](https://piripuz.github.io/Reverse_ADL_Vickrey/vickrey/likelihood/generate_data.html#vickrey.likelihood.generate_data.generate_arrival),
which, given the parameters and the number of wanted data,
returns the array of the optimized arrival times.

### Likelihood Optimization

To compute and optimize the likelihood, a variety of functions are available.

Functions in the module [`vickrey.likelihood.likelihood`](https://piripuz.github.io/Reverse_ADL_Vickrey/vickrey/likelihood/likelihood.html) allow the computation of the likelihood,
for [a single data point](https://piripuz.github.io/Reverse_ADL_Vickrey/vickrey/likelihood/likelihood.html#vickrey.likelihood.likelihood.likelihood) or for [a whole dataset](https://piripuz.github.io/Reverse_ADL_Vickrey/vickrey/likelihood/likelihood.html#vickrey.likelihood.likelihood.total_log_lik).

The module [`vickrey.likelihood.optimize`](https://piripuz.github.io/Reverse_ADL_Vickrey/vickrey/likelihood/optimize.html), lastly,
contains the functions used to optimize the likelihood.
The function [`optim_cycle`](https://piripuz.github.io/Reverse_ADL_Vickrey/vickrey/likelihood/optimize.html#vickrey.likelihood.optimize.optim_cycle) performs a grid search,
and once found the best initial conditions, runs an iterative optimizer with the given initialization.
The remaining functions can be used if more granularity in the control of the optimization is needed.
