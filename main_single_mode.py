import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt

import copy # to copy lists for python2.7

try:
    from .individual_single_mode import Individual
except Exception: #ImportError
    from individual_single_mode import Individual


############################### EVALFLAG.PY ##########################################
## A method to evaluate the quality of flagging.

## Inputs : data, flags
## Output : some metric of quality.

## Test : Run tfcrop/rflag with diff params, saving flagversions
##           Extract the 2D flags (perhaps just from flagversions)

VIS_DIR = '/Users/bjuncklaus/Dropbox/Evolutionary/Data/datasets/'
VIS_FILENAME = 'FewScans_G55_Spw2_Scan50.ms'
# VIS_FILENAME = 'FewScans_G55_Spw6_Scan6_HanningSmooth.ms'
# VIS_FILENAME = 'Four_ants_3C286.ms_Spw9_Scan30.ms'
# VIS_FILENAME = 'G29_Spw0_Scan145.ms'
# VIS_FILENAME = 'G29_Spw0_Scan38.ms'
# VIS_FILENAME = 'G29_Spw7_Scan83.ms'
# TODO - put the % Flagged on the best fit
# VIS_FILENAME = 'FewScans_G55_Spw6_Scan52_HanningSmooth.ms'
# VIS = '/home/vega2/bmartins/datasets/FewScans_G55_Spw2_Scan4.ms'
# VIS = '/Users/bjuncklaus/Dropbox/Evolutionary/Data/datasets/FewScans_G55_Spw6_Scan4_HanningSmooth.ms'
VIS = VIS_DIR + VIS_FILENAME

def runtest(cmdlist=[]):
    ## G55 dataset
    ## scans 4,6 are 3C286
    ## scans 50,52 are G55 SNR
    ## spws 6 and 11 have good representative RFI.


    # vis = '/home/vega2/bmartins/datasets/FewScans_G55_Spw7_Scan4.ms'

    # vis = '../Data/Four_ants_3C286.ms'

    spw = '0'
    scan = '4'
    vname = 'v1'

    ## Run the flagging
    flagdata(vis=VIS, mode='unflag', flagbackup=False)

    if (not cmdlist):
    #     tfcrop_params = "spw='" + spw + "' scan='" + scan + "'" + cmdlist[0]
    #     extend_params = "spw='" + spw + "' scan='" + scan + "'" + cmdlist[1]
    #     cmdlist = [tfcrop_params, extend_params]
    # else:
        # cmdlist = ["spw='"+spw+"' scan='"+scan+"' mode='tfcrop' maxnpieces=4 freqcutoff=3.0 usewindowstats='sum' " ,
        #              "spw='"+spw+"' scan='"+scan+"' mode='extend' growaround=True growtime=60.0" ]
        cmdlist = ["' mode='tfcrop'"]  # default value

    print()
    print("CMDLIST:", cmdlist)

    flagdata(vis=VIS, mode='list', inpfile=cmdlist, flagbackup=False)
    # flagmanager(vis=vis, mode='save', versionname=vname)

    ## Read the flags
    # flagmanager(vis=vis, mode='restore', versionname=vname)
    dat = getvals(col='DATA', vis=VIS)
    flag = getvals(col='FLAG', vis=VIS)

    # plotit(dat, flag)

    flag_percentage = np.sum(flag) / (1.0 * np.prod(flag.shape)) * 100.0
    print('% Flagged : ', flag_percentage)
    print('VIS : ', VIS_FILENAME)

    score = calcquality(dat, flag)
    if (math.isnan(score)):
        return float("inf"), flag_percentage

    return score, flag_percentage


def calcquality(dat, flag):
    """ Need to minimize the score that it returns"""

    shp = dat.shape

    npts = 0
    sumsq = 0.0
    maxval = 0.0
    leftover = []
    flagged = []
    for chan in range(0, shp[1]):
        for tm in range(0, shp[2]):
            val = np.abs(dat[0, chan, tm])
            if flag[0, chan, tm] == False:
                leftover.append(val)
            else:
                flagged.append(val)

    dmax, dmean, dstd = printstats(np.abs(dat[0, :, :]))
    rmax, rmean, rstd = printstats(leftover)
    fmax, fmean, fstd = printstats(flagged)

    maxdev = (rmax - rmean) / rstd
    fdiff = fmean - rmean
    sdiff = fstd - rstd

    print("Max deviation after flagging : ", maxdev)
    print("Diff in mean of flagged and unflagged : ", fdiff)
    print("Std after flagging : ", rstd)

    aa = np.abs(np.abs(maxdev) - 3.0)
    bb = 1.0 / ((np.abs(fdiff) - rstd) / rstd)
    cc = 1.0 / (np.abs(sdiff) / rstd)
    dd = 0.0

    pflag = (len(flagged) / (1.0 * shp[1] * shp[2])) * 100.0
    #
    # if pflag > 95.0:  # Check if what's flagged really looks like RFI.
    #     ## Mean and std should look similar...
    #     dd = (fmean - fstd) / fstd

    if pflag > 75.0:  # Check if what's flagged really looks like RFI.
        ## More flags means a worse score...
        dd = (pflag - 75.0) / 10.0

    res = np.sqrt(aa ** 2 + bb ** 2 + cc * 2 + dd * 2)

    if (fdiff < 0.0):
        res = res + res + 10.0

    print("Score : ", res)

    return res


def printstats(arr):
    if (len(arr) == 0):
        return 0, 0, 1

    med = np.median(arr)
    std = np.std(arr)
    maxa = np.max(arr)
    mean = np.mean(arr)
    # print 'median : ', med
    # print 'std : ', std
    # print 'max : ', maxa
    # print 'mean : ', mean
    # print " (Max - mean)/std : ", ( maxa - mean ) / std

    return maxa, mean, std

def getvals(col='DATA', vis="", spw="", scan=""):

    # print("SPW:", spw, "DDID:", ddid)

    tb.open(vis)
    if (spw and scan):
        tb.open(vis + '/DATA_DESCRIPTION')
        spwids = tb.getcol('SPECTRAL_WINDOW_ID')
        ddid = str(np.where(spwids == eval(spw))[0][0])
        tb1 = tb.query('SCAN_NUMBER==' + scan + ' && DATA_DESC_ID==' + ddid + ' && ANTENNA1=1 && ANTENNA2=2')
    else:
        tb1 = tb.query('ANTENNA1=1 && ANTENNA2=2')
    dat = tb1.getcol(col)
    tb1.close()
    tb.close()
    return dat

def plotit(dat, flag):
    pl.clf()
    pl.subplot(121)
    pl.imshow(np.abs(dat[0, :, :]))
    pl.subplot(122)
    pl.imshow(np.abs(dat[0, :, :] * (1 - flag[0, :, :])))



###############################################################################################################

# TODO - Fix evalflag

MODE = "rflag"
# MODE = "rflag"
MUTATION_PROB = 0.030
DROPOUT_PROB = 0.25
ELIMINATION_PROB = 0.4
# MUTATION_PROB = random.uniform(0.025, 0.05)
POPULATION_NUMBER = 20
TOTAL_GENERATION = 20

DISPLAY_STEP = 1
SAVE_OUTPUT = True

LOG_DIR = "/Users/bjuncklaus/Dropbox/Evolutionary/Data/results/"
LOG_FILENAME = LOG_DIR + time.strftime("%Y%m%d-%H%M%S") + "best_fit" + VIS_FILENAME.replace('/', '') + ".txt"

mutation_count = 0

population = []
children = []
fitness = []
best_fitness_historic = []

best_fitness_population = float("inf")
best_cmdlist_population = []

def initialize_params():
    params = []

    if (MODE == "tfcrop"):
        ## TFCROP ##
        # timecutoff, freqcutoff, maxnpieces, usewindowstats
        variation = 0.5
        timecutoff = [(i + 1) * variation for i in range(int(5/variation))]
        freqcutoff = [(i + 1) * variation for i in range(int(5/variation))]
        maxnpieces = [(i + 1) for i in range(7)]
        usewindowstats = ["std", "sum"]

        params.append(timecutoff)
        params.append(freqcutoff)
        params.append(maxnpieces)
        params.append(usewindowstats)
    else:
        ## RFLAG ##
        # timedevscale, freqdevscale, winsize
        variation = 0.5
        timedevscale = [(i + 1) * variation for i in range(int(5 / variation))]
        freqdevscale = [(i + 1) * variation for i in range(int(5 / variation))]
        winsize = [(i + 1) for i in range(5)]

        params.append(timedevscale)
        params.append(freqdevscale)
        params.append(winsize)

    ## EXTEND ##
    # growtime, growfreq, flagneartime, flagnearfreq
    variation = 10.0
    growtime = [((i + 1) * variation) + 40 for i in range(6)] # 50 - 100
    growfreq = [((i + 1) * variation) + 60 for i in range(4)] # 70 - 100
    flagneartime = [True, False]
    flagnearfreq = [True, False]
    growaround = [True, False]

    params.append(growtime)
    params.append(growfreq)
    params.append(flagneartime)
    params.append(flagnearfreq)
    params.append(growaround)

    print(params)

    return params

PARAMS = initialize_params()

def random_individual():
    individual = Individual(MODE)
    individual.set_fitness("inf")
    for param in PARAMS:
        random_index = random.randint(0, len(param) - 1)
        individual.add_attribute(param[random_index])

    return individual

def generate_individual():
    individual = random_individual()
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

    child1 = Individual(MODE, child1_attributes)
    child2 = Individual(MODE, child2_attributes)

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
                temp_attribute = copy.copy(PARAMS[random_index]) # using copy for python2.7
                temp_attribute.remove(old_attribute)

                new_random_index = random.randint(0, len(temp_attribute)-1)
                individual.update_attribute(random_index, temp_attribute[new_random_index])
            # print("After mutation: ", individual)

def elimate_random():
    global population
    for i in range(POPULATION_NUMBER):
        if (random.random() <= ELIMINATION_PROB):
            population[i] = random_individual()

def fitness_detail(function, fitness):
    min_max = function(fitness)
    random_droupout = random.random()
    if (random_droupout <= DROPOUT_PROB):
        fitness.remove(min_max)

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
    global best_fitness_population
    global best_cmdlist_population
    count = 1
    for individual in population:
        # TODO - Call CASA script

        print("Evaluating {0} out of {1}".format(count, POPULATION_NUMBER*2)) # The population doubles with the children
        count += 1

        fit, flag_percentage = runtest(individual.cmdlist())
        individual.set_fitness(fit)
        individual.set_flag_percentage(flag_percentage)
        fitness.append(individual.fitness)

        if (individual.fitness < best_fitness_population):
            # TODO - Check if it works
            best_fitness_population = individual.fitness
            best_cmdlist_population = individual.cmdlist()

        # SIMULATING FITNESS
        # if (individual.fitness < 0):
        #     random_fitness = random.random()
        #     individual.set_fitness(random_fitness)
        # fitness.append(individual.fitness)
        # END OF SIMULATION

    new_population, best_fitness = populate_best_individuals(fitness, save_fitness)

    population = copy.copy(new_population) # using copy for python2.7

    return best_fitness

def save_results(generation, max_individual, min_individual):
    best_fitness_historic.append(best_fitness_population)

    file = open(LOG_FILENAME, "a")

    file.write("Generation: " + str(generation+1))
    file.write("\n")
    file.write(max_individual.__str__())
    file.write("\n")
    file.write(min_individual.__str__())
    file.write("\n")
    file.write("############# Best Fitness of All: #############")
    file.write("\n")
    file.write(str(best_fitness_population))
    file.write("\n")
    file.write(str(best_cmdlist_population))
    # file.write(best_individual.__str__())
    # file.write("\n")
    # file.write(str(best_individual.cmdlist()))
    file.write("\n\n")

    file.close()

def show_results(generation, best_fitness):
    max_fitness, max_fitness_index = fitness_detail(np.min, best_fitness) # The smaller the best
    min_fitness, min_fitness_index = fitness_detail(np.max, best_fitness)

    print("##### RESULTS #####")
    print(mutation_count, "individuals have been mutated so far.")
    print("The fittest individual has a fitness of", max_fitness)
    max_individual = population[max_fitness_index]
    print("Info:", max_individual.__str__()) # using __str()__ for python2.7
    print()

    print("The least fit individual has a fitness of", min_fitness)
    min_individual = population[min_fitness_index]
    print("Info:", min_individual.__str__()) # using __str()__ for python2.7
    print()

    print("The best fitness of all generations is:", best_fitness_population)
    print(best_cmdlist_population)
    print("####################")

    if (SAVE_OUTPUT):
        save_results(generation, max_individual, min_individual)

def reset_global_variables():
    global fitness
    global children

    fitness = []
    children = []

def show_evolution_graph():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Fitness Evolution - ' + VIS_FILENAME + " Mode: " + MODE)
    ax.set_xlabel('generation')
    ax.set_ylabel('best fitness')
    ax.plot(range(len(best_fitness_historic)), best_fitness_historic, marker='o', color='b', linestyle='-')
    plt.show()

generate_population()

# Erasing previous content of the file
if (SAVE_OUTPUT):
    file = open(LOG_FILENAME, "w")
    file.write("Population: " + str(POPULATION_NUMBER))
    file.write("\n")
    file.close()

for generation in range(TOTAL_GENERATION):
    print("Current Generation:", generation + 1, "There are", TOTAL_GENERATION-(generation+1), "generations left")
    display_results = (generation+1) % DISPLAY_STEP == 0

    i = 0
    while i in range(POPULATION_NUMBER):
        crossover((population[i], population[i+1]))
        i += 2

    population = population + children

    mutate()

    elimate_random()

    best_fitness = evaluate_population(fitness, save_fitness=display_results)

    if (display_results):
        show_results(generation, best_fitness)

    reset_global_variables()

print("Finished")
show_evolution_graph()