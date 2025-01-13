import numpy as np
import test_functions

a = 1e-10
random_seed = 1234
np.random.seed(random_seed)


def generate_population(size, dimensions, lower_limit, upper_limit):
    return np.random.uniform(lower_limit, upper_limit, (size, dimensions))


def fitness_function(test_function, x):  # for minimization
    return -test_function(x)


def evaluate_population(population, test_function):
    return np.array([fitness_function(test_function, ind) for ind in population])


def choose_midpoint(population, evaluation, method):
    midpoint = None
    if len(population) > 0:
        if method == "mean":
            return np.mean(population, axis=0)
        elif method == "median":
            midpoint = np.median(population, axis=0)
        elif method == "weighted_mean":
            midpoint = np.average(population, axis=0, weights=evaluation)
        elif method == "weighted_geometric_mean":
            pass    # do djupy jest ta srednia bo ani wagi nie moga byc ujemne a potem jak juz sie to obejdzie to sa ujemne wartosci w punktach i znowu nie mozna wziac pierwiastkas
        elif method == "trimmed_mean" and len(population) > 3:
            sorted_population = np.sort(population, axis=0)
            limit = int(0.25 * len(population))  # classic trimmed mean doesn't include 25% of the lowest and highest outliers
            midpoint = np.mean(sorted_population[limit:-limit], axis=0)
        elif method == "huber_estimator":
            pass    # lowkey idk
    return midpoint


def onepoint_crossover(population, crossover_rate):
    if len(population) > 0:
        if len(population[0]) > 1:
            indices = np.arange(len(population))
            np.random.shuffle(indices)  # random parent pairing
            new_population = []
            for i in range(0, len(indices) - 1, 2):  # in case population is odd
                parent1 = population[indices[i]]
                parent2 = population[indices[i + 1]]
                if np.random.rand() < crossover_rate:
                    point = np.random.randint(1, len(parent1))
                    child1 = np.concatenate((parent1[:point], parent2[point:]))
                    child2 = np.concatenate((parent2[:point], parent1[point:]))
                else:
                    child1, child2 = parent1, parent2
                new_population.extend([child1, child2])
            if len(indices) % 2 == 1:  # if population is odd
                new_population.append(population[indices[-1]])
            return np.array(new_population)
        else:
            return population
    else:
        return population


def gaussian_mutation(population, mutation_rate, variance):  # during lectures variance was 0.04 but we didn't write this in initial report
    mutated_population = []
    for individual in population:
        if np.random.rand() < mutation_rate:
            mutation = np.random.normal(0, variance, size=len(individual))
            individual += mutation
        mutated_population.append(individual)
    return np.array(mutated_population)


def proportional_selection(population, evaluation):
    new_population = []
    adjustment = min(evaluation)
    if len(population) > 0:
        probabilities = (evaluation + adjustment + a) / (evaluation + adjustment + a).sum()     # probabilities can't be negative
        indices = np.random.choice(len(population), size=len(population), p=probabilities)
        new_population = []
        for index in indices:
            new_population.append(population[index])
    return np.array(new_population)


def genetic_algorithm(
    midpoint_method, test_function, population_lower_limit, population_upper_limit,
    dimensions=2, population_size=100, crossover_rate=0.7, mutation_rate=0.01,
    variance=0.04, epsilon=1e-10, max_generations=10000
):
    population = generate_population(population_size, dimensions, population_lower_limit, population_upper_limit)
    generation = 0
    evaluation = evaluate_population(population, test_function)
    midpoint = choose_midpoint(population, evaluation, midpoint_method)
    old_midpoint_evaluation = 1000
    if len(midpoint) > 0:
        new_midpoint_evaluation = fitness_function(test_function, midpoint)
        while (abs(old_midpoint_evaluation - new_midpoint_evaluation) > epsilon and (generation <= max_generations)):  # i had done something wrong or it doesnt work
            new_population = proportional_selection(population, evaluation)  # selection
            new_new_population = onepoint_crossover(new_population, crossover_rate)  # reproduction
            population = gaussian_mutation(np.array(new_new_population), mutation_rate, variance)  # mutation and generational succession - old population after reproduction and mutation becomes new population

            evaluation = evaluate_population(population, test_function)  # evaluation
            midpoint = choose_midpoint(population, evaluation, midpoint_method)
            old_midpoint_evaluation = new_midpoint_evaluation
            new_midpoint_evaluation = fitness_function(test_function, midpoint)

            generation += 1

    return population, midpoint, generation


population, midpoint, generation = genetic_algorithm("mean", test_functions.sphere_function, -5.12, 5.12)
evaluation = evaluate_population(population, test_functions.sphere_function)
best_individual = population[np.argmax(evaluation)]
midpoint_fitness = fitness_function(test_functions.sphere_function, midpoint)
print(generation)
print("Best individual:", best_individual)
print("Best fitness:", np.max(evaluation))
print("Midpoint:", midpoint)
print("Midpoint fitness:", midpoint_fitness)
