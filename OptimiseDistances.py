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
    :x:           Is a flattened array of all input vectors (array).
    :n_points:    The number of points to be placed in the space(int).
    :dim:         The dimension of the space in which we want to place points (int).

    :returns:           The minimum euclidean distance of all input vectors.
    """

    x = np.reshape(x, (dim, n_points)).transpose()
    combinations = list(it.combinations(x, 2))  # we list all the unique combinations of points
    norms = np.zeros(len(combinations))

    # take the norm for all combinations
    for index, comb in enumerate(combinations):
        norms[index] = (np.linalg.norm(comb[0] - comb[1]))

    return -np.amin(norms)  # the minimum norm of all points


def define_bounds(n_points, dim, upper_bound, lower_bound):
    """
    function that describes the bounds of your n-dimensional points

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
        bounds = np.array([(lower_bound, upper_bound) for n in range(n_points*dim)])

    elif len(lower_bound) == dim and len(upper_bound) == dim:
        bounds = np.array([(lower_bound[n], upper_bound[n]) for n in range(len(lower_bound)) for points in range(n_points)])

    else:
        print("wrong input for error bounds, will terminate")
        sys.exit()

    return bounds


def minimisation(n_points, dim, bounds, lower, upper):
    """
    driver function for minimisation

    :n_points (int):    The number of points to be placed in the space.
    :dim (int):         The dimension of the space in which we want to place points.

    :returns:           res:    Contains the result of the optimisation
                                    (scipy.optimize.OptimizeResult object).
                        new_points: contains the points obtained after convergence
                                (Numpy array, shape(n_points, dim).
    """
    if type(lower) == list:
        init_guess = np.array([np.random.uniform(bound[0], bound[1]) for bound in bounds])
    else:
        init_guess = np.random.uniform(lower, upper, size=(n_points * dim))
        print("yus")

    res = minimize(
        objective_function,
        x0=init_guess,
        args=(n_points, dim),
        bounds=bounds,
    )
    new_points = np.reshape(res.x, (dim, n_points))
    fun_value = -res.fun

    return fun_value, res, new_points


def run_n_optimisations(dim, n_points, upper, lower, n_optimisations):
    """
    function that runs the optimisation n-times to check for reasonable convergence
    and returns the vectors corresponding to the run with the largest euclidean distance

    Parameters:
    :n_points:      The number of points to be placed in the space (int).
    :dim:           The dimension of the space in which we want to place points (int).
    :upper:         The upper bound either for all dimensions as float or as array containing
                        a float for the particular bound of each dimension (float/array)
    :lower:         The lower bound either for all dimensions as float or as array containing
                        a float for the particular bound of each dimension (float /array)
    :n_optimisations:   The number of times you want to run the optimisation procedure

    :returns:       results_df: The distances to which the optimisation converged
                    point_vectors: the vectors with the best result on all runs
    """

    bounds = define_bounds(n_points, dim, upper, lower)

    run_results = np.zeros(n_optimisations)
    largest_distance = 0
    final_vectors = np.zeros((8, 7))

    for run in range(n_optimisations):

        fun_value, res, new_points = minimisation(n_points, dim, bounds, lower, upper)
        run_results[run] = fun_value

        # check if the current run has a larger distance than our previous best run
        # if yes save the vectors from from that run
        if largest_distance < -res.fun:
            largest_distance = -res.fun
            final_vectors = new_points

        print("Run number: " + str(run))
        print(final_vectors)
        print(res)

    results_df = pd.DataFrame(run_results)

    return final_vectors, results_df


def plot_histogram(results_df):
    """ plot the result of our runs as a histogram"""

    fig, ax = plt.subplots()
    results_df.plot(kind="hist", alpha=0.65, bins=25, ax=ax)
    ax.set_xlabel("Converged euclidean distance")
    ax.set_ylabel("Frequency")
    ax.set_title("Converged distances n = " + str(n_optimisations))
    ax.get_legend().remove()
    ax.tick_params(left=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    # plt.savefig('hist.png')
    plt.show()


# Driver code
if __name__ == '__main__':
    """PUT YOUR PARAMETERS HERE"""
    # n_points = 8
    # dim = 7
    # these parameters come from hanahs planet generator
    # upper = [2.1, 6.5, .65, 9, .5, 600, 0.85]
    # lower = [.4, .2, .12, .8, .05, 10, 0.15]
    n_points = 2
    dim = 2
    upper = 10
    lower = 0
    n_optimisations = 20
    bounds = define_bounds(n_points, dim, upper, lower)
    # fun_value, res, new_points = minimisation(n_points, dim, bounds)
    final_vectors, results_df = run_n_optimisations(dim, n_points, upper, lower, n_optimisations)
    plot_histogram(results_df)


