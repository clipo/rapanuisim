#!/usr/bin/env python
# Copyright (c) 2013.  Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Apache Software License, Version 2.0.  See the file LICENSE for details.

"""
the purpose here is to run a multi-scale spatially structured model for a fixed length of time, with innovation_rateation, and see what
happens to the structure of trait trees, sampled at fixed intervals.
Our goal is to look at the the rate at which  traits go extinct - measuring richeness and diversity.

"""
import simuOpt
import rapanuisim.utils as utils
simuOpt.setOptions(alleleType='long', optimized=True, quiet=False)

import logging as log
from time import time
import sys
import argparse
import uuid
import numpy.random as npr
import random
from rapanuisim import math as cpm
import rapanuisim
import rapanuisim.data as data
import rapanuisim.sampling as sampling

import rapanuisim.ct as ct

global config, sim_id, script, cores

def setup(parser):
    config = parser.parse_args()
    sim_id = uuid.uuid1().urn
    script = __file__

    if config.debug == '1':
        log.basicConfig(level=log.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
    else:
        log.basicConfig(level=log.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    # set up parallelism.  At the moment, this doesn't do anything on OSX
    # but should provide parallelism on Linux across the replicates at least

    if config.cores is not None:
        cores = config.cores
    else:
        cores = int(utils.get_parallel_cores(config.devel))
    log.info("Setting up %s cores for parallel simulation", cores)


    import simuOpt
    if(config.debug == 1):
        simuOpt.setOptions(alleleType='long',optimized=True,quiet=False,numThreads = cores)
    else:
        simuOpt.setOptions(alleleType='long',optimized=True,quiet=False,numThreads = cores)

    return (config,sim_id,script, cores)


def main():

    start = time()
    MAXALLELES = 10000000

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", help="provide name for experiment", required=True, type=str)
    parser.add_argument("--cores", type=int, help="Number of cores to use for simuPOP, overrides devel flag and auto calculation")
    parser.add_argument("--debug", help="turn on debugging output")
    parser.add_argument("--devel", help="Use only half of the available CPU cores", type=int, default=1)
    parser.add_argument("--dbhost", help="database hostname, defaults to localhost", default="localhost")
    parser.add_argument("--dbport", help="database port, defaults to 27017", default="27017")
    parser.add_argument("--stepsize", help="size of sample by proporition", type=float, default=1.0)
    parser.add_argument("--reps", help="Replicated populations per parameter set", type=int, default=4)
    parser.add_argument("--networkfile", help="Name of GML file representing the  network model for this simulation",
                        required=True, type=str)
    parser.add_argument("--numloci", help="Number of loci per individual", type=int, required=True)
    parser.add_argument("--maxinittraits", help="Max initial number of traits per locus for initialization", type=int,
                        required=True)
    parser.add_argument("--samplefraction", help="Size of samples taken to calculate all statistics, as a proportion", type=float,
                        required=True)
    parser.add_argument("--innovrate", help="Rate at which innovations occur in population as a per-locus rate", type=float, required=True)
    parser.add_argument("--simlength", help="Time at which simulation and sampling end, defaults to 3000 generations",
                        type=int, default="3000")
    parser.add_argument("--popsize", help="Initial size of population for each community in the model", type=int, required=True)
    parser.add_argument("--migrationfraction", help="Fraction of population that migrates each time step", type=float, required=True, default=0.2)
    parser.add_argument("--seed", type=int, help="Seed for random generators to ensure replicability")

    (config, sim_id, script, cores) = setup(parser)

    log.info("config: %s", config)

    beginCollectingData = cpm.expectedIAQuasiStationarityTimeHaploid(config.popsize, config.innovrate)
    log.info("Starting data collection at generation: %s", beginCollectingData)
    ### NOTE ###
    ###
    ### the simuPOP module is deliberately imported here because we need to process the
    ### command line arguments first, to understand which version of the simuPOP module (e.g.,
    ### long allele representation, etc, to import, and because we need to figure out how
    ### many cores the machine has, etc., to set it up for parallel processing.  If we import
    ### at the top of the file as normal, the imports happen before any code is executed,
    ### and we can't set those options.  DO NOT move these imports out of setup and main.
    import simuPOP as sim
    import demography as demo

    log.info("Starting simulation run %s", sim_id)
    log.debug("config: %s", config)
    if config.seed is None:
        log.info("No random seed given, allowing RNGs to initialize with random seed")
    else:
        log.debug("Seeding RNGs with seed: %s", config.seed)
        npr.seed(config.seed)
        random.seed(config.seed)

    full_command_line = " ".join(sys.argv)

    # Calculate the burn in time

    burn_time = rapanuisim.utils.simulation_burnin_time(config.popsize, config.innovrate)
    log.info("Minimum burn in time given popsize and theta: %s", burn_time)

    initial_distribution = rapanuisim.utils.constructUniformAllelicDistribution(config.maxinittraits)
    log.debug("Initial allelic distribution (for each locus): %s", initial_distribution)

    #innovation_rate = pypopgen.wf_mutation_rate_from_theta(config.popsize, config.innovrate)
    innovation_rate = float(config.innovrate)
    log.debug("Per-locus innov rate within populations: %s", innovation_rate)


    # Construct a demographic model from a collection of network slices which represent a temporal network
    # of changing subpopulations and interaction strengths.  This object is Callable, and simply is handed
    # to the mating function which applies it during the copying process
    networkmodel = demo.TemporalNetwork( networkmodel_path=config.networkfile,
                                         simulation_id=sim_id,
                                         sim_length=config.simlength,
                                         burn_in_time=burn_time,
                                         initial_subpop_size=config.popsize,
                                         migrationfraction=config.migrationfraction)

    # The regional network model defines both of these, in order to configure an initial population for evolution
    # Construct the initial population

    pop = sim.Population(size = networkmodel.get_initial_size(),
                         subPopNames = networkmodel.get_subpopulation_names(),
                         infoFields = networkmodel.get_info_fields(),
                         ploidy=1,
                         loci=config.numloci)

    log.info("population sizes: %s names: %s", pop.subPopSizes(), pop.subPopNames())
    initial_distribution = utils.constructUniformAllelicDistribution(config.numloci)
    log.info("Initial allelic distribution: %s", initial_distribution)


    # We are going to evolve the same population over several replicates, in order to measure how stochastic variation
    # effects the measured copying process.
    simu = sim.Simulator(pop, rep=config.reps)

    # Start the simulation and evolve the population, taking samples after the burn-in time has elapsed
    simu.evolve(
        initOps=sim.InitGenotype(freq=initial_distribution),
        preOps=[
            sim.PyOperator(func=sampling.logGenerationCount, param=(), step=100, reps=0)
        ],
        matingScheme=sim.RandomSelection(subPopSize=networkmodel),
        postOps=[sim.KAlleleMutator(k=MAXALLELES, rates=config.innovrate, loci=sim.ALL_AVAIL),
                 sim.PyOperator(func=rapanuisim.data.sampleNumAlleles,
                                param=(config.samplefraction, config.innovrate, config.popsize, sim_id, config.numloci),
                                step=config.stepsize, begin=beginCollectingData),
                 sim.PyOperator(func=rapanuisim.data.sampleTraitCounts,
                                param=(config.samplefraction, config.innovrate, config.popsize, sim_id, config.numloci),
                                step=config.stepsize, begin=beginCollectingData),
                 sim.PyOperator(func=rapanuisim.data.censusTraitCounts,
                                param=(config.innovrate, config.popsize, sim_id, config.numloci), step=config.stepsize,
                                begin=beginCollectingData),
                 sim.PyOperator(func=rapanuisim.data.censusNumAlleles,
                                param=(config.innovrate, config.popsize, sim_id, config.numloci), step=config.stepsize,
                                begin=beginCollectingData),
                 sim.PyOperator(func=rapanuisim.data.sampleIndividuals,
                                param=(config.samplefraction, config.innovrate, config.popsize, sim_id, config.numloci),
                                step=config.stepsize, begin=beginCollectingData),

        ],
        gen=config.simlength,

    )

    endtime = time()
    elapsed = endtime - start
    #log.info("simulation complete in %s seconds with %s cores", elapsed, cores)
    log.info("simulation complete,%s,%s",cores,elapsed)
    log.info("Ending simulation run at generation %s", simu.population(0).dvars().gen)
    sampled_length = int(config.simlength) - burn_time

if __name__ == "__main__":
    main()





