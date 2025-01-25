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

#plot_function(sphere_function, (-5.12, 5), (-5.12, 5.12), "Sphere function")
plot_function(eggholder_function, (-512, 512), (-512, 512), "Eggholder function", midpoints[0], bestpoints[0]) #ploting for first iteration
#plot_function(rosenbrock_function, (-5, 10), (-5, 10), "Rosenbrock function")
