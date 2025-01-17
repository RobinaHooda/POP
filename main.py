import numpy as np
from genetic_algorithm import evaluate_population, fitness_function, genetic_algorithm
import test_functions
from midpoint import midpoint_mean


def main():
    population, midpoint, generation = genetic_algorithm(midpoint_mean, test_functions.sphere_function, -5.12, 5.12)
    evaluation = evaluate_population(population, test_functions.sphere_function)
    best_individual = population[np.argmax(evaluation)]
    midpoint_fitness = fitness_function(test_functions.sphere_function, midpoint)
    print(generation)
    print("Best individual:", best_individual)
    print("Best fitness:", np.max(evaluation))
    print("Midpoint:", midpoint)
    print("Midpoint fitness:", midpoint_fitness)


if __name__ == "__main__":
    main()
