import numpy as np


a = 1e-12
random_seed = 1234
np.random.seed(random_seed)


def generate_population(size, dimensions, domain):
    dimensions = len(domain)
    population = np.empty((size, dimensions))

    for i, (lower_limit, upper_limit) in enumerate(domain):
        population[:, i] = np.random.uniform(lower_limit, upper_limit, size)

    return population


def fitness_individual(individual, test_function, domain):
    for i, (lower_limit, upper_limit) in enumerate(domain):
        if individual[i] < lower_limit or individual[i] > upper_limit:
            return 10000
    return test_function(individual)


def fitness_function(population, test_function, domain):
    scores = np.empty(len(population))
    for i, individual in enumerate(population):
        scores[i] = fitness_individual(individual, test_function, domain)
    return scores


def evaluate_population(population, scores):
    best = population[0]
    best_score = scores[0]

    for i, individual in enumerate(population):
        if scores[i] < best_score:
            best = individual
            best_score = scores[i]
    return best, best_score


def onepoint_crossover(population, crossover_rate):
    if not len(population):
        return population

    new_population = []
    while len(population) > 1:
        indices = np.random.choice(len(population), size=2, replace=False)
        parent1, parent2 = population[indices[0]], population[indices[1]]

        population = np.delete(population, indices, axis=0)
        if np.random.uniform(0, 1) < crossover_rate:
            point = np.random.randint(1, len(parent1))
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
        else:
            child1, child2 = parent1, parent2
        new_population.extend([child1, child2])

    if len(population) == 1:
        new_population.append(population[0])

    return new_population


def gaussian_mutation(population, mutation_rate, variance):
    mutated_population = []
    for individual in population:
        mutation_mask = np.random.uniform(0, 1, size=individual.shape) < mutation_rate
        mutation = np.random.normal(0, variance, size=len(individual)) * mutation_mask
        mutated = individual + mutation
        mutated_population.append(mutated)
    return np.array(mutated_population)


def convert_scores_to_weights(scores):
    adjustment = min(scores) + a
    return 1 / (scores + adjustment)


def proportional_selection(population, scores):
    new_population = []

    probabilities = convert_scores_to_weights(scores)
    probabilities /= probabilities.sum()        # normalization
    indices = np.random.choice(len(population), size=len(population), p=probabilities)

    for index in indices:
        new_population.append(population[index])
    return np.array(new_population)


def genetic_algorithm(
    choose_midpoint, test_function, domain,
    dimensions=2, population_size=100, crossover_rate=0.7, mutation_rate=0.1,
    variance=0.04, epsilon=1e-10, max_generations=10000
):
    # mutation_rate was changed from 0.01 to 0.1
    # mutation_variance was set to 0.04
    population = generate_population(population_size, dimensions, domain)
    generation = 0

    scores = fitness_function(population, test_function, domain)
    best, best_score = evaluate_population(population, scores)

    midpoint = choose_midpoint(population, scores)
    old_midpoint_evaluation = 1000

    new_midpoint_evaluation = fitness_individual(midpoint, test_function, domain)
    while (
        abs(old_midpoint_evaluation - new_midpoint_evaluation) > epsilon and
        (generation <= max_generations)
    ):
        population = proportional_selection(population, scores)                         # selection
        population = onepoint_crossover(population, crossover_rate)                     # crossover
        population = gaussian_mutation(np.array(population), mutation_rate, variance)   # mutation

        scores = fitness_function(population, test_function, domain)
        new_best, new_best_score = evaluate_population(population, scores)
        if new_best_score < best_score:
            best = new_best
            best_score = new_best_score

        midpoint = choose_midpoint(population, scores)
        old_midpoint_evaluation = new_midpoint_evaluation
        new_midpoint_evaluation = fitness_individual(midpoint, test_function, domain)

        generation += 1

    return population, midpoint, generation-1, best, best_score
