#!/usr/bin/env python
# Copyright (c) 2013.  Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Apache Software License, Version 2.0.  See the file LICENSE for details.

from __future__ import print_function
import simuOpt, sys
simuOpt.setOptions(alleleType='long',optimized=False,quiet=True)
import simuPOP as sim
import uuid
import rapanuisim.data as data
import rapanuisim.utils as utils
import rapanuisim.math as cpm
import ming
import logging as log
import pprint as pp
import argparse

"""
This program simulates the Wright-Fisher model of genetic drift with infinite-alleles mutation in a single
population, and counts the number of alleles present in the population and in samples of specified size
"""

# get all parameters
args = argparse.ArgumentParser()
args.add_argument("--popsize",
                  default=1000,
                  type=int,
                  help="Population Size")
args.add_argument("--stepsize",
                  default=100,
                  type=int,
                  help="Interval Between Data Samples")
args.add_argument("--length",
                  default=10000,
                  type=int,
                  help="Length of simulation sample (in generations) after stationarity reached")
args.add_argument("--replications",
                  default=5,
                  type=int,
                  help="Number of populations to simulate in parallel")
args.add_argument("--mutationrate",
              default=0.001,
              type=float,
              help="Rate of individual innovations/mutations per generation (i.e., noise factor)")
args.add_argument("--numloci",
              default=30,
              type=int,
              help="Number of loci to model (number of features)")
args.add_argument("--states",
              default=10,
              type=int,
              help="Number of states each locus or feature can take")
args.add_argument("--experiment_name",
              default="ct_",
              help="Name of experiment to prefix database tables")
args.add_argument("--samplesize",
                  default=30,
                  type=int,
                  help="Size of sample to take each generation for allele counting.")
args = args.parse_args()

# we're not loading a config file here, taking defaults
config_file = None
simconfig = utils.CTPyConfiguration(config_file)


log.basicConfig(level=log.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

log.debug("experiment name: %s", args.experiment_name)
log.debug("NOTE:  This interactive simulation always sends data to MongoDB instance on localhost")

data.set_experiment_name(args.experiment_name)
data.set_database_hostname("localhost")
data.set_database_port("27017")
config = data.getMingConfiguration()
ming.configure(**config)

sim_id = uuid.uuid4().urn


log.info("Beginning simulation run: %s", sim_id)


beginCollectingData = cpm.expectedIAQuasiStationarityTimeHaploid(args.popsize,args.mutationrate)
log.info("Starting data collection at generation: %s", beginCollectingData)

totalSimulationLength = beginCollectingData + args.length
log.info("Simulation will sample %s generations after stationarity", args.length)

data.storeSimulationData(args.popsize,args.mutationrate,sim_id,args.samplesize,args.replications,args.numloci,__file__,args.numloci,simconfig.MAXALLELES)

initial_distribution = utils.constructUniformAllelicDistribution(args.numloci)
log.info("Initial allelic distribution: %s", initial_distribution)

pop = sim.Population(size=args.popsize, ploidy=1, loci=args.numloci)
simu = sim.Simulator(pop, rep=args.replications)

simu.evolve(
	initOps = sim.InitGenotype(freq=initial_distribution),
    preOps = [
        sim.PyOperator(func=utils.logGenerationCount, param=(), step=1000, reps=0),
    ],
	matingScheme = sim.RandomSelection(),
	postOps = [sim.KAlleleMutator(k=simconfig.MAXALLELES, rates=args.mutationrate, loci=sim.ALL_AVAIL),
        sim.PyOperator(func=data.sampleNumAlleles, param=(args.samplesize, args.mutationrate, args.popsize,sim_id,args.numloci), step=args.stepsize,begin=beginCollectingData),
        sim.PyOperator(func=data.sampleTraitCounts, param=(args.samplesize, args.mutationrate, args.popsize,sim_id,args.numloci), step=args.stepsize,begin=beginCollectingData),
        sim.PyOperator(func=data.censusTraitCounts, param=(args.mutationrate, args.popsize,sim_id,args.numloci), step=args.stepsize,begin=beginCollectingData),
        sim.PyOperator(func=data.censusNumAlleles, param=(args.mutationrate, args.popsize,sim_id,args.numloci), step=args.stepsize,begin=beginCollectingData),
        #sim.PyOperator(func=data.sampleIndividuals, param=(args.samplesize, args.mutationrate, args.popsize, sim_id,args.numloci), step=args.stepsize, begin=beginCollectingData),
		],	
	gen = totalSimulationLength,
)

log.info("Ending simulation run at generation %s", simu.population(0).dvars().gen)



