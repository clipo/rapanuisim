from __future__ import division
from collections import defaultdict, OrderedDict
from copy import deepcopy
import simuOpt
simuOpt.setOptions(alleleType='long', optimized=True, quiet=False)
import math
import seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt
import demography.network as network
import simuPOP as sp
from simuPOP import demography
import logging as log
import numpy as np
import scipy.stats
from matplotlib import colors as mcolors

## now set up the basic parameters of the simulation (need to change this to a config file...)
num_loci = 1        ## for now we just need one
pop_size = 5000
num_gens = 6000
migs = [0.001, 0.01, 0.1]
pop_list = [100, 500, 1000]
innovation_rate = 0.00
MAXALLELES = 1000
connectedness=3 ## k
sub_pops=25
migration_fraction=0.5
num_starting_alleles=1000
divisor = 100.0 / num_starting_alleles
frac = divisor / 100.0
distribution = [frac] * num_starting_alleles
save_state=True
burn_in_time = 4000

output={}

k_values=[2,6,10,20,25]
run_param=k_values
for k in run_param:
    output[k]=[]

### some functions to store stats at each timestep.
def init_acumulators(pop, param):
    acumulators = param
    for acumulator in acumulators:
        if acumulator.endswith('_sp'):
            pop.vars()[acumulator] = defaultdict(list)
        else:
            pop.vars()[acumulator] = []
            pop.vars()['allele_frequencies'] = []
            pop.vars()['haplotype_frequencies'] = []
            pop.vars()['allele_count']=[]
            pop.vars()['richness'] = []
            pop.vars()['class_freq']=[]
            pop.vars()['class_count']=[]
            #pop.vars()['fst_mean']
    return True

def update_acumulator(pop,  param):
    acumulator, var = param
    #for acumulator, var in sorted(param.items()):
    log.debug("acumulator: %s var: %s" % (acumulator,var))
    if  var.endswith('_sp'):
        for sp in range(pop.numSubPop()):
            pop.vars()[acumulator][sp].append(deepcopy(pop.vars(sp)[var[:-3]]))
    else:
        pop.vars()[acumulator].append(deepcopy(pop.vars()[var]))
        #pop.vars()['richness'].append(deepcopy(pop.vars()['haploFreq']))
        #output[run_param].append(deepcopy(pop.vars()[var]))
    return True

def update_richness_acumulator(pop,  param):
    (acumulator, var) = param
    if  var.endswith('_sp'):
        for sp in range(pop.numSubPop()):
            pop.vars()[acumulator][sp].append(len(pop.dvars(sp).alleleFreq[0].values()))
    else:
        pop.vars()['haplotype_frequencies'].append(len(pop.dvars().haploFreq.values()))
        pop.vars()['allele_frequencies'].append(len(pop.dvars().alleleFreq.values()))
        pop.vars()['allele_count'].append(len(pop.dvars().alleleNum))
        #pop.vars()[acumulator].append(deepcopy(pop.vars()[var]))
    return True

def sampleAlleleAndGenotypeFrequencies(pop,  param):
    (popsize, num_loci) = param

    sp.stat(pop, haploFreq = range(0, num_loci), vars=['haploFreq', 'haploNum'])
    #sim.stat(pop, alleleFreq = sim.ALL_AVAIL)

    keys = list(pop.dvars().haploFreq.keys())
    haplotype_map = pop.dvars().haploFreq[keys[0]]
    haplotype_count_map = pop.dvars().haploNum[keys[0]]
    num_classes = len(haplotype_map)

    #class_freq = {'-'.join(i[0]) : str(i[1]) for i in haplotype_map.items()}
    class_freq = dict()
    for k,v in haplotype_map.items():
        key = '-'.join(str(x) for x in k)
        class_freq[key] = v
    #log.debug("class_freq packed: %s", class_freq)

    class_count = dict()
    for k,v in haplotype_count_map.items():
        key = '-'.join(str(x) for x in k)
        class_count[key] = v
    pop.vars()['richness'].append(num_classes)
    pop.vars()['class_freq'].append(class_freq)
    pop.vars()['class_count'].append(class_count)
    return True

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

iteration_number=-1
for param_value in run_param:
    iteration_number += 1
    ## these are lists of things that simuPop will do at different stages
    init_ops = OrderedDict()
    pre_ops = OrderedDict()
    post_ops = OrderedDict()

    # Construct a demographic model from a collection of network slices which represent a temporal network
    # of changing subpopulations and interaction strengths.  This object is Callable, and simply is handed
    # to the mating function which applies it during the copying process
    #networkmodel = NetworkModel( networkmodel="/Users/clipo/Documents/PycharmProjects/RapaNuiSim/notebooks/test_graph.gml",
    networkmodel = network.NetworkModel( networkmodel="smallworld",
                                         simulation_id="1",
                                         sim_length=3000,
                                         burn_in_time=burn_in_time,
                                         initial_subpop_size=pop_size,
                                         migrationfraction=migration_fraction,
                                         sub_pops=sub_pops,
                                         connectedness=param_value, # if 0, then distance decay
                                         save_figs=save_state,
                                         network_iteration=iteration_number)

    num_pops = networkmodel.get_subpopulation_number()
    sub_pop_size = int(pop_size / num_pops)

    # The regional network model defines both of these, in order to configure an initial population for evolution
    # Construct the initial population
    pops = sp.Population( size = networkmodel.get_initial_size(),
                         subPopNames = str(list(networkmodel.get_subpopulation_names())),
                         infoFields = 'migrate_to',
                         ploidy=1,
                         loci=100 )

    ### now set up the activities
    init_ops['acumulators'] = sp.PyOperator(init_acumulators, param=['fst','alleleFreq', 'haploFreq'])
    init_ops['Sex'] = sp.InitSex()
    init_ops['Freq'] = sp.InitGenotype(loci=0,freq=distribution)
    post_ops['Innovate']=sp.KAlleleMutator(k=MAXALLELES, rates=innovation_rate, loci=sp.ALL_AVAIL)
    #post_ops['mig'] = sp.Migrator(demography.migrIslandRates(migration_rate, num_pops)) #, reps=[i])
    post_ops['mig']=sp.Migrator(rate=networkmodel.get_migration_matrix())
    #for i, mig in enumerate(migs):
    #        post_ops['mig-%d' % i] = sp.Migrator(demography.migrIslandRates(mig, num_pops), reps=[i])

    post_ops['Stat-fst'] =sp.Stat(structure=sp.ALL_AVAIL)
    #post_ops['haploFreq']=sp.stat(pops, haploFreq=[0], vars=['haploFreq', 'haploNum'])
    #post_ops['alleleFreq']=sp.stat(pops, alleleFreq=sp.ALL_AVAIL)

    post_ops['Stat-richness']=sp.Stat(alleleFreq=[0], haploFreq=[0], vars=['alleleFreq','haploFreq','alleleNum', 'genoNum'])
    post_ops['fst_acumulation'] = sp.PyOperator(update_acumulator, param=['fst','F_st'])
    post_ops['richness_acumulation'] = sp.PyOperator(update_richness_acumulator, param=('alleleFreq', 'Freq of Alleles'))
    post_ops['class_richness']=sp.PyOperator(sampleAlleleAndGenotypeFrequencies, param=(pop_size,num_loci))

    mating_scheme = sp.RandomSelection()
    #mating_scheme=sp.RandomSelection(subPopSize=sub_pop_size)

    ## go simuPop go! evolve your way to the future!
    sim = sp.Simulator(pops) #, rep=3)

    sim.evolve(initOps=list(init_ops.values()), preOps=list(pre_ops.values()), postOps=list(post_ops.values()),
               matingScheme=mating_scheme, gen=num_gens)

    print("now evolving... k= %s" % param_value)

    # now make a figure of the Fst results
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    for pop, mig in zip(sim.populations(), migs):
        ax.plot(pop.dvars().fst, label='Migration rate %.4f' % mig)
    ax.legend(loc=2)
    ax.set_ylabel('FST')
    ax.set_xlabel('Generation')
    plt.show()

    # now make a figure of the Fst results
    fig1 = plt.figure(figsize=(16, 9))
    ax = fig1.add_subplot(111)
    for pop, pop_size_example in zip(sim.populations(), pop_list):
        ax.plot(pop.dvars().fst, label='Population Size %s' % pop_size_example)
    ax.legend(loc=2)
    ax.set_ylabel('FST')
    ax.set_xlabel('Generation')
    plt.show()

    #print (pop.dvars())
    # # now make a figure of the alleleFreq results...
    fig2 = plt.figure(figsize=(16, 9))
    ax = fig2.add_subplot(111)
    for pop, mig in zip(sim.populations(), migs):
        ax.plot(pop.dvars().richness, label='Migration rate %.4f' % mig)
    ax.legend(loc=2)
    ax.set_ylabel('Richness')
    ax.set_xlabel('Generation')
    plt.show()
    output[param_value]=deepcopy(pop.dvars())
    #print(output)

sum_fig = plt.figure(figsize=(16,9))
ax=sum_fig.add_subplot(111)
iteration=-1
for k in run_param:
    iteration += 1
    ax.plot(output[k].fst, color=list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())[iteration], label='k - %s' % k)
ax.legend(loc=2)
ax.set_ylabel('Fst')
ax.set_xlabel('Generations')
plt.show()
sum_fig.savefig('sum_fig.png', bbox_inches='tight')

rich_fig = plt.figure(figsize=(16,9))
ax=rich_fig.add_subplot(111)
iteration=-1
for k in run_param:
    iteration+=1
    ax.plot(output[k].richness, color=list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())[iteration], label='k - %s' % k)
ax.legend(loc=2)
ax.set_ylabel('Richness')
ax.set_xlabel('Generations')
plt.show()
rich_fig.savefig('richness.png', bbox_inches='tight')

print("k value      mean            lower                upper")
for k in run_param:
    print("k - %s: %s" % (k, mean_confidence_interval(output[k].fst[4000:8000], confidence=0.95)))

# # now make a figure of the haplotypeFreq results...
# fig3 = plt.figure(figsize=(16, 9))
# ax = fig3.add_subplot(111)
# for pop, mig in zip(sim.populations(), migs):
#     ax.plot(pop.dvars().alleleNum, label='Migration rate %.4f' % mig)
# ax.legend(loc=2)
# ax.set_ylabel('Allele Numbers')
# ax.set_xlabel('Generation')
# plt.show()
# #
# # now make a figure of the haplotypeFreq results...
# fig4 = plt.figure(figsize=(16, 9))
# ax = fig4.add_subplot(111)
# for pop, mig in zip(sim.populations(), migs):
#     ax.plot(pop.dvars().class_richness, label='Migration %.4f' % mig)
# ax.legend(loc=2)
# ax.set_ylabel(' Class Richness')
# ax.set_xlabel('Generation')
# plt.show()