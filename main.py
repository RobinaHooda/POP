from genetic_algorithm import fitness_individual, genetic_algorithm
import test_functions
from midpoint import midpoint_mean


def main():
    _, midpoint, generation, best, best_score = genetic_algorithm(
        choose_midpoint=midpoint_mean,
        test_function=test_functions.sphere_function,
        domain=[(-5.12, 5.12), (-5.12, 5.12)]
    )
    midpoint_fitness = fitness_individual(
        midpoint,
        test_functions.sphere_function,
        [(-5.12, 5.12), (-5.12, 5.12)]
    )
    print("Generations:", generation)
    print("Best individual:", best)
    print("Best fitness score:", best_score)
    print("Midpoint:", midpoint)
    print("Midpoint fitness score:", midpoint_fitness)


if __name__ == "__main__":
    main()
