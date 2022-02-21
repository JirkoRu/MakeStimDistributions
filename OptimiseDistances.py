import numpy as np
import itertools as it
from scipy.optimize import minimize
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""
code intended to make maximally distant points in n-dimensional space, 
to be used for generative stimuli
Author: Jirko Rubruck
"""


def objective_function(x, n_points, dim):
    """
    objective function to maximise the minimal euclidean distance
    between a set of points in an n-dimensional space.

    Parameters:
    :x:         Is a flattened array of all input vectors (array).
    :n_points:    The number of points to be placed in the space(int).
    :dim:         The dimension of the space in which we want to place points (int).

    :returns:           The minimum euclidean distance of all input vectors.
    """

    x = np.reshape(x, (n_points, dim))
    combinations = list(it.combinations(x, 2))  # we list all the unique combinations of points
    norms = np.zeros(len(combinations))

    # take the norm for all combinations
    for index, comb in enumerate(combinations):
        norms[index] = (np.linalg.norm(comb[0] - comb[1]))

    return -np.amin(norms)  # the minimum of all points


def define_bounds(n_points, dim, upper_bound, lower_bound):
    """
    function that decscribe the bounds of your n-dimensional points

    Parameters:
    :n_points:      The number of points to be placed in the space (int).
    :dim:           The dimension of the space in which we want to place points (int).
    :upper_bound:   The upper bound either for all dimensions as float or as array containing
                        a float for the particular bound of each dimension (float/array)
    :lower_bound:   The lower bound either for all dimensions as float or as array containing
                        a float for the particular bound of each dimension (float /array)

    :returns:           the bounds for each variable in the objective function. (list of tuples)
    """

    if isinstance(lower_bound, (int, float)) and isinstance(upper_bound, (int, float)):
        bounds = [(lower_bound, upper_bound) for n in range(n_points*dim)]

    elif len(lower_bound) == dim and len(upper_bound) == dim:
        bounds = [(lower_bound[n], upper_bound[n]) for n in range(len(dim))]

    else:
        print("wrong input for error bounds, will terminate")
        sys.exit()

    return bounds


def minimisation(n_points,dim,bounds):
    """
    driver function for minimisation

    :n_points (int):    The number of points to be placed in the space.
    :dim (int):         The dimension of the space in which we want to place points.

    :returns:           res:    Contains the result of the optimisation
                                    (scipy.optimize.OptimizeResult object).
                        new_points: contains the points obtained after convergence
                                (Numpy array, shape(n_points, dim).
    """

    res = minimize(
        objective_function,
        x0=np.random.random(n_points * dim),
        args=(n_points, dim),
        bounds=bounds,
    )
    new_points = np.reshape(res.x, (n_points, dim))
    fun_value = -res.fun

    return fun_value, res, new_points


def run_n_optimisations(dim,n_points, upper, lower, n_optimisations):
    bounds = define_bounds(n_points, dim, upper, lower)
    run_results = np.zeros(n_optimisations)
    for run in range(n_optimisations):
        fun_value, res, new_points = minimisation(n_points, dim, bounds)
        run_results[run] = fun_value
        print(run)
    results_df = pd.DataFrame(run_results)
    fig, ax = plt.subplots()
    # results_df.plot(kind="kde", ax=ax, color="lightcoral")
    # results_df.plot(kind="hist", density=True, alpha=0.65, bins=25, ax=ax)
    results_df.plot(kind="hist", alpha=0.65, bins=25, ax=ax)
    ax.set_xlabel("Converged euclidean distance")
    ax.set_ylabel("Frequency")
    ax.set_title("Converged distances n = " + str(n_optimisations))
    ax.get_legend().remove()
    ax.tick_params(left=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    plt.show()
    plt.savefig('hist2.png')


# Driver code
if __name__ == '__main__':
    n_points = 8
    dim = 7
    upper = 5
    lower = 0
    n_optimisations = 200
    # bounds = define_bounds(n_points, dim, upper, lower)
    # fun_value, res, new_points = minimisation(n_points, dim, bounds)
    # print(res)
    # print(new_points)
    run_n_optimisations(dim, n_points, upper, lower, n_optimisations)
