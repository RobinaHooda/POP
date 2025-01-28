import json
import numpy as np
from test_functions import sphere_function, rosenbrock_function, eggholder_function, plot_function

midpoints = []
bestpoints = []
generations = []
best_individuals = []
with open("results/genetic_algorithm_results.json", 'r') as file:
    data = json.load(file)
for entry in data:
    midpoints.append(entry["midpoints"])
    bestpoints.append(entry["best_individuals"])
    generations.append(entry["generations"])
    best_individuals.append(entry["best_individual"])

test_functions = [
    ["Sphere Function", sphere_function, (-5.12, 5.12), (-5.12, 5.12), [[0, 0]]],
    ["Rosenbrock Function", rosenbrock_function, (-5, 10), (-5, 10), [[1, 1]]],
    ["EggHolder Function", eggholder_function, (-512, 512), (-512, 512), [[512, 404.2319]]],
]
midpoint_functions = ["Mean", "Median", "Trimmed Mean", "Weighted Mean", "Weighted Geometric Mean"]

results = []
for test_function_no, test_function in enumerate(test_functions):
    for midpoint_function_no, midpoint_function in enumerate(midpoint_functions):
        range_start = (test_function_no * len(midpoint_functions) + midpoint_function_no) * 30
        range_end = (test_function_no * len(midpoint_functions) + midpoint_function_no + 1) * 30
        mean_generations = np.mean(generations[range_start:range_end])
        real_point = np.array(test_function[4][0])
        distances = np.linalg.norm(np.array(best_individuals[range_start:range_end]) - real_point, axis=1)
        mean_distance = np.mean(distances)

        result_entry = {
            "function": test_function[0],
            "midpoint_function": midpoint_function,
            "domain": [test_function[2], test_function[3]],
            "real_result": test_function[4][0],
            "mean_distance": mean_distance,
            "mean_generations": mean_generations
        }
        results.append(result_entry)

        print(f"Function: {test_function[0]}, midpoint: {midpoint_function}, domain: [{test_function[2]}, {test_function[3]}], real_result: {test_function[4][0]}")
        print(f"Mean distance of best individual from real result: {mean_distance}")
        print(f"Mean generations: {mean_generations}\n")
        plot_function(test_function[1], test_function[2], test_function[3], f"{test_function[0]} {midpoint_function}", True, midpoints[range_start:range_end], bestpoints[range_start:range_end], test_function[4])
    print("==================================================================\n")
with open("results/summary.json", "w") as outfile:
    json.dump(results, outfile, indent=4)