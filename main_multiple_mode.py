import random
import numpy as np

try:
    from .individual_multiple_mode import Individual
except Exception: #ImportError
    from individual_multiple_mode import Individual

MUTATION_PROB = 0.025
# MUTATION_PROB = random.uniform(0.025, 0.05)
POPULATION_NUMBER = 100
TOTAL_GENERATION = 10000

DISPLAY_STEP = 100

mutation_count = 0

population = []
children = []
fitness = []

def initialize_params():
    # TODO - Change params
    mode = ['rflag', 'tfcrop']
    freqdevscale = [i + 1 for i in range(5)]
    flagneartime = [True, False]
    flagnearfreq = [True, False]

    variation = 10
    growtime = [(i + 1) * variation for i in range(8)]
    growfreq = [(i + 1) * variation for i in range(8)]

    return [mode, freqdevscale, flagneartime, flagnearfreq, growtime, growfreq]

PARAMS = initialize_params()

def generate_individual():

    individual = Individual()
    for param in PARAMS:
        random_index = random.randint(0, len(param)-1)
        individual.add_attribute(param[random_index])
    population.append(individual)

def generate_population():
    for _ in range(POPULATION_NUMBER):
        generate_individual()

def combine(individual1, individual2, cut_index):
    chromossome1 = individual1.get_attributes()[0:cut_index]
    chromossome2 = individual1.get_attributes()[cut_index:len(individual1.get_attributes())]

    chromossome3 = individual2.get_attributes()[0:cut_index]
    chromossome4 = individual2.get_attributes()[cut_index:len(individual2.get_attributes())]

    child1_attributes = []
    child2_attributes = []

    child1_attributes.extend(chromossome1) # First half of individual1
    child1_attributes.extend(chromossome4) # Last half of individual2

    child2_attributes.extend(chromossome3) # First half of individual2
    child2_attributes.extend(chromossome2) # Last half of individual1

    child1 = Individual(child1_attributes)
    child2 = Individual(child2_attributes)

    return child1, child2

def crossover(individuals, crosspoint=2):
    individual1 = individuals[0]
    individual2 = individuals[1]

    cut_index = int(len(PARAMS) / crosspoint)

    child1, child2 = combine(individual1, individual2, cut_index) # Creating children

    global children
    children.append(child1)
    children.append(child2)

def mutate():
    for individual in population:
        if (random.random() < MUTATION_PROB):
            global mutation_count
            mutation_count += 1
            # print("Before mutation: ", individual)
            for _ in range(2):
                random_index = random.randint(0, len(PARAMS)-1)
                old_attribute = individual.get_attributes()[random_index]
                temp_attribute = PARAMS[random_index].copy()
                temp_attribute.remove(old_attribute)

                new_random_index = random.randint(0, len(temp_attribute)-1)
                individual.update_attribute(random_index, temp_attribute[new_random_index])
            # print("After mutation: ", individual)

def fitness_detail(function, fitness):
    min_max = function(fitness)
    return (min_max, fitness.index(min_max))

def populate_best_individuals(fitness, save_fitness):
    best_fitness = []
    best_individuals = []
    for _ in range(POPULATION_NUMBER):
        max_fitness, max_fitness_index = fitness_detail(np.min, fitness)

        if (save_fitness):
            best_fitness.append(max_fitness) # Saving best fitness

        best_individuals.append(population[max_fitness_index]) # Saving best individual for the new generation

        fitness.remove(max_fitness)

    return (best_individuals, best_fitness)

def evaluate_population(fitness, save_fitness=False):
    global population
    for individual in population:
        # TODO - Call CASA script

        # SIMULATING FITNESS
        fitness.append(random.random())
        # END OF SIMULATION
        continue

    new_population, best_fitness = populate_best_individuals(fitness, save_fitness)

    population = new_population.copy()

    return best_fitness

def show_results(generation, best_fitness):
    max_fitness, max_fitness_index = fitness_detail(np.min, best_fitness) # The smaller the best (if using RMS)
    min_fitness, min_fitness_index = fitness_detail(np.max, best_fitness)

    print(mutation_count, "individuals (total of {0}%) have been mutated".format("{:.2f}".format((mutation_count/(POPULATION_NUMBER*generation*2)) * 100)))
    print("The fittest individual has a fitness of", max_fitness)
    print("Info:", population[max_fitness_index])

    print("The least fit individual has a fitness of", min_fitness)
    print("Info:", population[min_fitness_index])

def reset_global_variables():
    global fitness
    global children

    fitness = []
    children = []

generate_population()

for generation in range(TOTAL_GENERATION):
    print("Current Generation:", generation, "There are", TOTAL_GENERATION-generation, "generations left")
    display_results = (generation+1) % DISPLAY_STEP == 0
    for i in range(POPULATION_NUMBER):
        if (i % 3 == 0):
            continue
        # print("len", len(population), i, i+1)
        crossover((population[i], population[i+1]))

    population = population + children

    mutate()

    best_fitness_population = evaluate_population(fitness, save_fitness=display_results)

    if (display_results):
        show_results(generation, best_fitness_population)

    reset_global_variables()


print("Population size:", len(population))