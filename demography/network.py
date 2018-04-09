#!/usr/bin/env python
# Copyright (c) 2015.  Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Apache Software License, Version 2.0.  See the file LICENSE for details.

"""
Description here

"""
import networkx as nx
import numpy as np
import re
from rapanuisim import math
import simuPOP as sim
import logging as log

class NetworkModel(object):
    """
    NetworkModel implements a full "demographic model" in simuPOP terms,
    that is, it manages the size and existence of subpopulations, and the
    migration matrix between them.  The basic data for this model is derived
    by importing a NetworkX network in the form of a GML format file.
    The network edges represent a set of subpopulations
    with unique ID's, and edges between them which are weighted.  The
    weights may be determined by any underlying model (e.g., distance,
    social interaction hierarchy, etc), but need to be interpreted here
    purely as the probability of individuals moving between two subpopulations,
    since that is the implementation of interaction.
    """

    def __init__(self,
                 networkmodel=None,
                 simulation_id=None,
                 sim_length=10000,
                 burn_in_time=0,
                 initial_subpop_size = 0,
                 migrationfraction = 0.2):
        """
        :param networkmodel: Name of GML file
        :param sim_length: Number of generations to run the simulation
        :return:
        """
        #BaseMetapopulationModel.__init__(self, numGens = sim_length, initSize = initial_size, infoFields = info_fields, ops = ops)
        self.network = networkmodel
        self.sim_length = sim_length
        self.info_fields = 'migrate_to'
        self.sim_id = simulation_id
        self.burn_in_time = burn_in_time
        self.init_subpop_size = initial_subpop_size
        self.migration_fraction = migrationfraction

        self.times = []

        self._cached_migration_matrix = None

        # This will be set when we examine the network model
        self.sub_pops = 0

        self.subpopulation_names = []

        # Parse the GML files and create a list of NetworkX objects
        self._parse_network_model()

        # Determine the order and time of network slices
        self._assign_slice_times()

        # Determine the initial population configuration
        self._calculate_initial_population_configuration()

        # prime the migration matrix
        self._cached_migration_matrix = self._calculate_migration_matrix(min(self.times))

    ############### Private Initialization Methods ###############

    def _parse_network_model(self):
        """
        Given a file,  read the GML files (format: <name>.gml)
        and construct aNetworkX networkmodel from the GML file
        """

        log.debug("Opening  GML file %s:",  self.networkmodel)
        gml = nx.read_gml(self.networkmodel)
        slice = nx.parse_gml(gml)
        #log.debug("slice nodes: %s", '|'.join(sorted(slice.nodes())))
        self.network = slice

    def _calculate_initial_population_configuration(self):
        # num subpops is just the number of vertices in the first graph slice.
        first_time = min(self.times)
        network = self.network
        self.sub_pops = network.number_of_nodes()
        log.debug("Number of initial subpopulations: %s", self.sub_pops)

        # subpoplation names - have to switch them to plain strings from unicode or simuPOP won't use them as subpop names
        self.subpopulation_names =  [d["label"].encode('utf-8', 'ignore') for n,d in network.nodes_iter(data=True)]
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

        g_mat = nx.to_numpy_matrix(self.network).astype(nx.float)
        print("g_mat: ", g_mat)
        # get the column totals
        rtot = np.sum(g_mat, axis = 1)
        scaled = (g_mat / rtot) * self.migration_fraction
        diag = np.eye(np.shape(g_mat)[0]) * (1.0 - self.migration_fraction)
        g_mat_scaled = diag + scaled

        log.debug("scaled migration matrix: %s", g_mat_scaled.tolist())

        return g_mat_scaled.tolist()


    ###################### Public API #####################

    def is_change_time(self, gen):
        return gen in self.times

    def get_info_fields(self):
        return self.info_fields

    def get_initial_size(self):
        return [self.init_subpop_size] * self.sub_pops

    def get_subpopulation_names(self):
        return self.subpopulation_names

    def get_subpopulation_sizes(self):
        return self.subpop_sizes



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

        sim.migrate(pop, self._cached_migration_matrix)
        sim.stat(pop, popSize=True)
        # cache the new subpopulation names and sizes for debug and logging purposes
        # before returning them to the calling function
        self.subpopulation_names = sorted(pop.subPopNames())
        self.subpop_sizes = pop.subPopSizes()
        return pop.subPopSizes()

