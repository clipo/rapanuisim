#!/usr/bin/env python
# Copyright (c) 2018 Carl P. Lipo <clipo@binghamton.edu> with much derived from the work of Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Apache Software License, Version 2.0.  See the file LICENSE for details.

"""
Description here

"""
import networkx as nx
import numpy as np
import simuPOP as sim
import logging as log


class NetworkModel(object):
    """
    RapaNui implements a full "demographic model" in simuPOP terms,
    that is, it manages the size and existence of subpopulations, and the
    migration matrix between them.  The basic data for this model is derived
    by importing a a NetworkX network model in the form of a GML format.
    The network represents subpopulations with unique ID's, and edges
    between them which are weighted.  The GML file is created in a separate process. The
    weights may be determined by any underlying model (e.g., distance,
    social interaction hierarchy, etc), but need to be interpreted here
    purely as the probability of individuals moving between two subpopulations,
    since that is the implementation of interaction.
    """

    def __init__(self,
                 networkfile=None,
                 simulation_id=None,
                 sim_length=10000,
                 burn_in_time=0,
                 initial_subpop_size = 0,
                 migrationfraction = 0.2):
        """
        :param networkmodel_path: List of full paths to a set of GML files
        :param sim_length: Number of generations to run the simulation
        :return:
        """
        #BaseMetapopulationModel.__init__(self, numGens = sim_length, initSize = initial_size, infoFields = info_fields, ops = ops)
        self.networkfile = networkfile
        self.sim_length = sim_length
        self.info_fields = 'migrate_to'
        self.sim_id = simulation_id
        self.burn_in_time = burn_in_time
        self.init_subpop_size = initial_subpop_size
        self.migration_fraction = migrationfraction


        self._cached_migration_matrix = None

        # This will be set when we examine the network model
        self.sub_pops = 0

        self.subpopulation_names = []

        # Parse the GML file and create a list of NetworkX objects
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
        Given a file,  read GML file and construct a  NetworkX networkmodeling
        """
        self.network_model = nx.read_gml(self.network_file, relabel=False)

        #log.debug("Map of slice_id to time: %s", self.sliceid_to_time_map)


    def _calculate_initial_population_configuration(self):
        # num subpops is just the number of vertices in the graph.

        self.sub_pops = self.network_model.number_of_nodes()
        #log.debug("Number of initial subpopulations: %s", self.sub_pops)

        # subpoplation names - have to switch them to plain strings from unicode or simuPOP won't use them as subpop names
        self.subpopulation_names =  [d["label"].encode('utf-8', 'ignore') for n,d in self.network_model.nodes_iter(data=True)]
        #log.debug("calc_init_config:  subpopulation names: %s", self.subpopulation_names)


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


    def _calculate_migration_matrix(self, time):
        g_cur = self.time_to_network_map[self.sliceid_to_time_map[self._get_sliceid_for_time(time)]]
        g_mat = nx.to_numpy_matrix(g_cur)

        # get the column totals
        rtot = np.sum(g_mat, axis = 1)
        scaled = (g_mat / rtot) * self.migration_fraction
        diag = np.eye(np.shape(g_mat)[0]) * (1.0 - self.migration_fraction)
        g_mat_scaled = diag + scaled

        log.debug("scaled migration matrix: %s", g_mat_scaled.tolist())

        return g_mat_scaled.tolist()



    ###################### Debug Methods ##################

    def _dbg_slice_pop_start(self,pop,time):
        """
        Debug method for comparing the assemblages in a network model slice versus the simuPOP population.
        """
        g_prev = self.time_to_network_map[self.sliceid_to_time_map[self._get_previous_sliceid_for_time(time)]]
        slice_str = '|'.join(sorted(g_prev.nodes()))
        pop_str = '|'.join(sorted(pop.subPopNames()))
        log.debug("start slice: %s", slice_str)
        log.debug("start smpop: %s", pop_str)


    def _dbg_slice_pop_end(self,pop,time):
        """
        Debug method for comparing the assemblages in a network model slice versus the simuPOP population.
        """
        g_cur = self.time_to_network_map[self.sliceid_to_time_map[self._get_sliceid_for_time(time)]]
        slice_str = '|'.join(sorted(g_cur.nodes()))
        pop_str = '|'.join(sorted(pop.subPopNames()))
        log.debug("end slice: %s", slice_str)
        log.debug("end smpop: %s", pop_str)



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

    def get_subpopulation_durations(self):
        """
        Returns a map with subpopulation name as key, and duration as value.  Burn-in time is NOT
        included in the duration, so the durations are relative to the simlength - burnin.
        """

        duration = dict()
        for sp in self.node_exit_time:
            duration[sp] = int(self.node_exit_time[sp]) - int(self.node_origin_time[sp])
            #log.debug("duration sp %s: %s - from %s to %s", sp, duration[sp], self.node_origin_time[sp], self.node_exit_time[sp])
        return duration

    def get_subpopulation_origin_times(self):
        origins = dict()
        origins.update(self.node_origin_time)
        return origins

    def get_subpopulation_slice_ids(self):
        slices = dict()
        slices.update(self.node_slice_map)
        return slices

    def __call__(self, pop):
        """
        Main public interface to this demography model.  When the model object is called in every time step,
        this method determines whether a new network slice is now active.  If so, the requisite changes
        to subpopulations are made (adding/deleting subpopulations), and then the new migration matrix is
        applied.

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


        # At the first time slice, start tracking duration for assemblages that exist at min(self.times)
        # which is also the end of the burn-in time
        # After this point, further changes are recorded as nodes are added/deleted
        if gen == min(self.times):
            starting_subpops = pop.subPopNames()
            for subpop in starting_subpops:
                self.node_origin_time[subpop] = min(self.times)

        # At the very end of the simulation, after the last slice time, we finish off the
        # duration times of assemblages that exist at sim_length.
        if gen == self.sim_length - 1:
            log.debug("End of simulation: recording exit time for assemblages present at sim_length")
            ending_subpops = pop.subPopNames()
            for subpop in ending_subpops:
                self.node_exit_time[subpop] = self.sim_length


        ######### Do the per tick processing ##########

        if self.is_change_time(gen) == False:
            pass
        else:
            slice_for_time = self.time_to_sliceid_map[gen]
            log.debug("========= Processing network slice %s at time %s =============", slice_for_time, gen)
            #self._dbg_slice_pop_start(pop,gen)


            # switch to a new network slice, first handling added and deleted subpops
            # then calculate a new migration matrix
            # then migrate according to the new matrix
            (added_subpops, deleted_subpops) = self._get_added_deleted_subpops_for_time(gen)

            # add new subpopulations
            #log.debug("time %s adding subpops: %s", gen, added_subpops)
            for sp in added_subpops:
                (origin_sp, origin_sp_name) = self._get_origin_subpop_for_new_subpopulation(gen,pop,sp)

                log.debug("time %s new node: %s parent id: %s  parent name: %s", gen, sp, origin_sp, origin_sp_name)

                sp_names = [origin_sp_name, sp]
                #log.debug("spnames: %s", sp_names)
                split_ids = pop.splitSubPop(origin_sp, [0.5, 0.5], sp_names)
                #log.debug("split return: %s", split_ids)
                # make sure all subpopulations are the same size, sampling from existing individuals with replacement
                numpops = pop.numSubPop()
                sizes = [self.init_subpop_size] * numpops
                pop.resize(sizes, propagate=True)

            # delete subpopulations
            #log.debug("time %s deleting subpops: %s", gen, deleted_subpops)
            for sp in deleted_subpops:
                #log.debug("pops: %s", pop.subPopNames())
                log.debug("time %s deleted subpop: %s", gen, sp)
                pop.removeSubPops(pop.subPopByName(sp))

            # update the migration matrix
            self._cached_migration_matrix = self._calculate_migration_matrix(gen)

            #self._dbg_slice_pop_end(pop,gen)


        sim.migrate(pop, self._cached_migration_matrix)
        sim.stat(pop, popSize=True)
        # cache the new subpopulation names and sizes for debug and logging purposes
        # before returning them to the calling function
        self.subpopulation_names = sorted(pop.subPopNames())
        self.subpop_sizes = pop.subPopSizes()
        return pop.subPopSizes()

