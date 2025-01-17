import numpy as np


def midpoint_mean(population, _):
    return np.mean(population, axis=0)


def midpoint_median(population, _):
    return np.median(population, axis=0)


def midpoint_weighted_mean(population, evaluation):
    return np.average(population, axis=0, weights=evaluation)


def midpoint_weighted_geometric_mean(population, evaluation):
    return np.exp(np.average(np.log(population), axis=0, weights=evaluation))


def midpoint_trimmed_mean(population, _):
    sorted_population = np.sort(population, axis=0)
    limit = int(0.25 * len(population))
    return np.mean(sorted_population[limit:-limit], axis=0)


def midpoint_huber_estimator(population, _):
    pass
