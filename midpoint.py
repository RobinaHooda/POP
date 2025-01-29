import numpy as np
from genetic_algorithm import convert_scores_to_weights
from scipy.optimize import minimize


def midpoint_mean(population, _):
    return np.mean(population, axis=0)


def midpoint_median(population, _):
    return np.median(population, axis=0)


def midpoint_weighted_mean(population, scores):
    return np.average(population, axis=0, weights=convert_scores_to_weights(scores))


def midpoint_weighted_geometric_mean(population, scores):
    weights = convert_scores_to_weights(scores)
    log_population = np.log(np.abs(population) + 1e-6)

    geometric_mean = np.exp(np.average(log_population, axis=0, weights=weights))
    if np.sum(np.sign(population)) <= 0:
        geometric_mean = -geometric_mean

    return geometric_mean


def midpoint_trimmed_mean(population, _):
    trim_fraction = 0.15
    sorted_population = np.sort(population, axis=0)
    limit = int(trim_fraction * len(population))
    return np.mean(sorted_population[limit:-limit], axis=0)


def midpoint_huber_estimator(population, _):
    def huber_loss(mu):
        delta = 1.0
        a = population - mu
        return np.sum(np.where(
            np.abs(a) <= delta,
            0.5 * a**2,
            delta * (np.abs(a) - 0.5 * delta)
        ))

    initial_guess = np.mean(population, axis=0)
    return minimize(huber_loss, initial_guess).x
