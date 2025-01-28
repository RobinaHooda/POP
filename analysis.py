import json
from test_functions import sphere_function, rosenbrock_function, eggholder_function, plot_function

midpoints = []
with open("results/genetic_algorithm_results.json", 'r') as file:
    data = json.load(file)
for entry in data:
    if "midpoints" in entry:
        midpoints.append(entry["midpoints"])
bestpoints = []
with open("results/genetic_algorithm_results.json", 'r') as file:
    data = json.load(file)
for entry in data:
    if "best_individuals" in entry:
        bestpoints.append(entry["best_individuals"])

test_functions = [
    ["Sphere Function", sphere_function, (-5.12, 5.12), (-5.12, 5.12), [0, 0]],
    ["Rosenbrock Function", rosenbrock_function, (-5, 10), (-5, 10), [1, 1]],
    ["EggHolder Function", eggholder_function, (-512, 512), (-512, 512), [512, 404.2319]],
]
midpoint_functions = ["Mean", "Median", "Trimmed Mean", "Weighted Mean", "Weighted Geometric Mean"]

for test_function_no, test_function in enumerate(test_functions):
    for midpoint_function_no, midpoint_function in enumerate(midpoint_functions):
        range_start = (test_function_no * len(midpoint_functions) + midpoint_function_no) * 30
        range_end = (test_function_no * len(midpoint_functions) + midpoint_function_no + 1) * 30 + 1
        plot_function(test_function[1], test_function[2], test_function[3], f"{test_function[0]} {midpoint_function}", True, midpoints[range_start:range_end], bestpoints[range_start:range_end], test_function[4])