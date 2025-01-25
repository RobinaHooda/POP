import numpy as np
import json
from genetic_algorithm import fitness_individual, genetic_algorithm
import test_functions
from midpoint import (
    midpoint_mean,
    midpoint_median,
    midpoint_trimmed_mean,
    midpoint_weighted_mean,
    midpoint_weighted_geometric_mean,
)


def main():
    test_functions_list = [
    #    ("Sphere", test_functions.sphere_function, [(-5.12, 5.12), (-5.12, 5.12)]),
    #    ("Rosenbrock", test_functions.rosenbrock_function, [(-5, 10), (-5, 10)]),
        ("EggHolder", test_functions.eggholder_function, [(-512, 512), (-512, 512)]),
    ]

    midpoint_functions = [
        ("Mean", midpoint_mean),
    #    ("Median", midpoint_median),
    #    ("Trimmed Mean", midpoint_trimmed_mean),
    #    ("Weighted Mean", midpoint_weighted_mean),
    #    ("Weighted Geometric Mean", midpoint_weighted_geometric_mean),
    ]

    n = 30
    results = []

    for test_name, function, domain in test_functions_list:
        print(f"\nRunning genetic algorithm for {test_name} function:\n")
        for midpoint_name, midpoint_fn in midpoint_functions:
            print(f"Using midpoint strategy: {midpoint_name}")
            for i in range(n):
                print(f"Iteration: {i}")
                np.random.seed(i + 1)
                midpoints, midpoint, midpoint_score, generation, best_individuals, best, best_score = genetic_algorithm(
                    choose_midpoint=midpoint_fn,
                    test_function=function,
                    domain=domain
                )
                result = {
					"function": test_name,
					"midpoint_strategy": midpoint_name,
					"seed": i + 1,
					"generations": generation,
                    "midpoints": [midpoint.tolist() for midpoint in midpoints],
					"midpoint": midpoint.tolist(),
					"midpoint_fitness_score": midpoint_score,
                    "best_individuals": [best.tolist() for best in best_individuals],
					"best_individual": best.tolist(),
					"best_fitness_score": best_score,
				}
                results.append(result)

    with open("results/genetic_algorithm_results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)


if __name__ == "__main__":
    main()
