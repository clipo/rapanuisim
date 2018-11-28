from __future__ import division
from collections import defaultdict, OrderedDict
from copy import deepcopy
import simuOpt
simuOpt.setOptions(alleleType='long', optimized=True, quiet=False)

import seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt

import simuPOP as sp
from simuPOP import demography
import networkx as nx
import logging as log
import numpy as np

## for now include this in this file - until we can figure out what it needs to do.
class NetworkModel(object):
    """
    NetworkModel implements a full "demographic model" in simuPOP terms,
    that is, it manages the size and existence of subpopulations, and the
    migration matrix between them.  The basic data for this model is derived
    by the creation of a random small-world NetworkX network in the (watts_strogatz_graph) or
    the creation of a network from a GML file.
    The network edges represent a set of subpopulations
    with unique ID's, and edges between them which are weighted.  The
    weights may be determined by any underlying model (e.g., distance,
    social interaction hierarchy, etc), but need to be interpreted here
    purely as the probability of individuals moving between two subpopulations,
    since that is the implementation of interaction.
    """

    def __init__(self,
                 networkmodel="smallworld",
                 simulation_id=None,
                 sim_length=1000,
                 burn_in_time=0,
                 initial_subpop_size = 0,
                 migrationfraction = 0.01,
                 sub_pops=5,
                 connectedness=3,
                 rewiring_prob=0.0):
        """
        :param networkmodel: Name of GML file
        :param sim_length: Number of generations to run the simulation
        :return:
        """
        #BaseMetapopulationModel.__init__(self, numGens = sim_length, initSize = initial_size, infoFields = info_fields,
        # ops = ops, sub_pops = num_subpops, connectedness = connectedness, rewiring_prob=rewiring_prob)
        self.networkmodel = networkmodel        #default is small world - else GML file location
        self.sim_length = sim_length
        self.info_fields = 'migrate_to'
        self.sim_id = simulation_id
        self.burn_in_time = burn_in_time
        self.init_subpop_size = initial_subpop_size
        self.migration_fraction = migrationfraction
        self.connectedness = connectedness  # default of 3
        self.sub_pops = sub_pops            # default of 5
        self.rewiring_prob = rewiring_prob  # default of 0.0

        self._cached_migration_matrix = None

        # This will be set when we examine the network model
        self.sub_pops = 0

        self.subpopulation_names = []

        # Parse the GML files and create a list of NetworkX objects
        self._parse_network_model()

        # Determine the initial population configuration
        self._calculate_initial_population_configuration()

        # prime the migration matrix
        self._cached_migration_matrix = self._calculate_migration_matrix()

    ############### Private Initialization Methods ###############

    def _parse_network_model(self):
        """
        Given a file,  read the GML files (format: <name>.gml)
        and construct aNetworkX networkmodel from the GML file
        """
        if (self.networkmodel=="smallworld"):
            log.debug("Creating small world Watts-Strogatz network")
            network = nx.watts_strogatz_graph(self.sub_pops, self.connectedness, 0)
            log.debug("network nodes: %s", '|'.join(sorted(network.nodes())))
            self.network = network
        else:
            log.debug("Opening  GML file %s:",  self.networkmodel)
            network = nx.read_gml(self.networkmodel)
            log.debug("network nodes: %s", '|'.join(sorted(network.nodes())))
            self.network = network

    def _calculate_initial_population_configuration(self):
        # num subpops is just the number of vertices in the first graph slice.
        #first_time = min(self.times)
        network = self.network
        self.sub_pops = network.number_of_nodes()
        log.debug("Number of initial subpopulations: %s", self.sub_pops)
        log.debug("list of nodes: %s", list(network.nodes(data=True)))
        # subpoplation names - have to switch them to plain strings from unicode or simuPOP won't use them as subpop names
        self.subpopulation_names =  list(network.nodes)

        log.debug("calc_init_config:  subpopulation names: %s", self.subpopulation_names)


    ############### Private Methods for Call() Interface ###############


    def _get_node_label(self,g, id):
        return g.node[id]["label"].encode('utf-8', 'ignore')


    def _get_id_for_subpop_name(self,pop,name):
        return pop.subPopByName(name)

    def _get_node_parent(self,g, id):
        return g.node[id]["parent_node"].encode('utf-8', 'ignore')


    def _get_subpop_idname_map(self, pop):
        names = pop.subPopNames()
        name_id_map = dict()
        for name in names:
            id = pop.subPopByName(name)
            name_id_map[id] = name
        return name_id_map


    def _calculate_migration_matrix(self):

        g_mat = nx.to_numpy_matrix(self.network).astype(np.float)
        print("g_mat: ", g_mat)
        # get the column totals
        rtot = np.sum(g_mat, axis = 1)
        scaled = (g_mat / rtot) * self.migration_fraction
        diag = np.eye(np.shape(g_mat)[0]) * (1.0 - self.migration_fraction)
        g_mat_scaled = diag + scaled
        log.debug("scaled migration matrix: %s", g_mat_scaled.tolist())
        return g_mat_scaled.tolist()


    ###################### Public API #####################

    def get_info_fields(self):
        return self.info_fields

    def get_initial_size(self):
        return [self.init_subpop_size] * self.sub_pops

    def get_subpopulation_names(self):
        return self.subpopulation_names

    def get_subpopulation_sizes(self):
        return self.subpop_sizes

    def get_subpopulation_number(self):
        return len(self.subpopulation_names)

    def __call__(self, pop):
        """
        Main public interface to this demography model.  When the model object is called in every time step,
        this method creates a new migration matrix.

        After migration, the stat function is called to inventory the subpopulation sizes, which are then
        returned since they're handed to the RandomSelection mating operator.

        If a new network slice is not active, the migration matrix from the previous step is applied again,
        and the new subpopulation sizes are returns to the RandomSelection mating operator as before.

        :return: A list of the subpopulation sizes for each subpopulation
        """
        if 'gen' not in pop.vars():
            gen = 0
        else:
            gen = pop.dvars().gen

        ######### Do the per tick processing ##########

        log.debug("========= Processing network  =============")
        #self._dbg_slice_pop_start(pop,gen)

        # update the migration matrix
        self._cached_migration_matrix = self._calculate_migration_matrix(gen)

        sp.migrate(pop, self._cached_migration_matrix)
        sp.stat(pop, popSize=True)
        # cache the new subpopulation names and sizes for debug and logging purposes
        # before returning them to the calling function
        self.subpopulation_names = sorted(pop.subPopNames())
        self.subpop_sizes = pop.subPopSizes()
        print(self.subpop_sizes)
        return pop.subPopSizes()


## now set up the basic parameters of the simulation (need to change this to a config file...)
num_loci = 10
pop_size = 5000
num_gens = 100

migs = [0.001, 0.01, 0.1]
innovation_rate = 0.01
MAXALLELES = 10000
divisor = 100.0 / num_loci
frac = divisor / 100.0
distribution = [frac] * num_loci

## these are lists of things that simuPop will do at different stages
init_ops = OrderedDict()
pre_ops = OrderedDict()
post_ops = OrderedDict()

### some functions to store stats at each timestep.
def init_acumulators(pop, param):
    acumulators = param
    for acumulator in acumulators:
        if acumulator.endswith('_sp'):
            pop.vars()[acumulator] = defaultdict(list)
        else:
            pop.vars()[acumulator] = []
    return True

def update_acumulator(pop, param):
    acumulator, var = param
    if  var.endswith('_sp'):
        for sp in range(pop.numSubPop()):
            pop.vars()[acumulator][sp].append(deepcopy(pop.vars(sp)[var[:-3]]))
    else:
        pop.vars()[acumulator].append(deepcopy(pop.vars()[var]))
    return True

# Construct a demographic model from a collection of network slices which represent a temporal network
# of changing subpopulations and interaction strengths.  This object is Callable, and simply is handed
# to the mating function which applies it during the copying process
networkmodel = NetworkModel( networkmodel="/Users/clipo/Documents/PycharmProjects/RapaNuiSim/notebooks/test_graph.gml",
                                     simulation_id="1",
                                     sim_length=3000,
                                     burn_in_time=500,
                                     initial_subpop_size=1000,
                                     migrationfraction=0.01)

num_pops = networkmodel.get_subpopulation_number()
sub_pop_size = int(pop_size / num_pops)

# The regional network model defines both of these, in order to configure an initial population for evolution
# Construct the initial population

pops = sp.Population(size = networkmodel.get_initial_size(),
                     subPopNames = list(networkmodel.get_subpopulation_names()),
                     infoFields = 'migrate_to',
                     ploidy=1,
                     loci=100 )


### now set up the activities
init_ops['acumulators'] = sp.PyOperator(init_acumulators, param=['fst'])
init_ops['Sex'] = sp.InitSex()
init_ops['Freq'] = sp.InitGenotype(freq=distribution)
post_ops['Innovate']=sp.KAlleleMutator(k=MAXALLELES, rates=innovation_rate, loci=sp.ALL_AVAIL)
for i, mig in enumerate(migs):
    post_ops['mig-%d' % i] = sp.Migrator(demography.migrIslandRates(mig, num_pops), reps=[i])
post_ops['Stat-fst'] = sp.Stat(structure=sp.ALL_AVAIL)
post_ops['fst_acumulation'] = sp.PyOperator(update_acumulator, param=('fst', 'F_st'))
mating_scheme = sp.RandomSelection()
#mating_scheme=sp.RandomSelection(subPopSize=sub_pop_size)

## go simuPop go!
sim = sp.Simulator(pops, rep=len(migs))
sim.evolve(initOps=list(init_ops.values()), preOps=list(pre_ops.values()), postOps=list(post_ops.values()),
           matingScheme=mating_scheme, gen=num_gens)

# now make a figure of the results
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
for pop, mig in zip(sim.populations(), migs):
    ax.plot(pop.dvars().fst, label='mig rate %.4f' % mig)
ax.legend(loc=2)
ax.set_ylabel('FST')
ax.set_xlabel('Generation')
plt.show()