#!/usr/bin/env python
# Copyright (c) 2013.  Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Apache Software License, Version 2.0.  See the file LICENSE for details.

from __future__ import print_function
import simuOpt, sys
simuOpt.setOptions(alleleType='long',optimized=False,quiet=True)
import simuPOP as sim
import uuid
import ctpy.data as data
import ctpy.utils as utils
import ctpy.math as cpm
import ming
import logging as log
import pprint as pp


"""
This program simulates the Wright-Fisher model of genetic drift with infinite-alleles mutation in a single
population, and counts the number of alleles present in the population and in samples of specified size
"""


options = [
    {
        'name':'popsize',
        'default':1000,
        'label':'Population Size',
        'type': 'integer',
        'validator': 'popsize > 0',
    },
    {
        'name':'stepsize',
        'default':100,
        'label':'Interval Between Data Samples',
        'type': 'integer',
        'validator': 'stepsize > 0',
    },
    {
        'name':'length',
        'default':10000,
        'label':'Length of simulation sample (in generations) after stationarity reached',
        'type': 'integer',
        'validator': 'length > 0',
    },
    {
        'name':'replications',
        'default':5,
        'label':'Number of populations to simulate in parallel',
        'type': 'integer',
        'validator': 'replications > 0',
    },
    {
        'name':'samplesize',
        'default':30,
        'label':'Size of sample to take each generation for allele counting',
        'type': 'integer',
        'validator': 'samplesize > 0',
    },
    {
        'name':'mutationrate',
        'default':0.001,
        'label':'Rate of individual innovations/mutations per generation',
        'type': 'number',
        'validator': 'mutationrate > 0.0',
    },
    {
        'name':'numloci',
        'default' : 2,
        'label' : 'Number of loci to model',
        'type' : 'integer',
        'validator' : 'numloci > 0',
    },
    {
        'name' : 'numalleles',
        'default' : 10,
        'label' : 'Number of initial alleles in population',
        'type' : 'integer',
        'validator' : 'numalleles > 0'
    },
    {
        'name' : 'experiment_name',
        'default' : 'default',
        'label' : 'Name of experiment to prefix database tables',
        'type' : 'string',
    }
]


# get all parameters
pars = simuOpt.Params(options, doc='This program simulates the Wright-Fisher model of genetic drift with infinite-alleles mutation in a single population, and counts the number of alleles present in the population and in samples of specified size')
# cancelled
if not pars.getParam():
    sys.exit(1)

# we're not loading a config file here, taking defaults
config_file = None
simconfig = utils.CTPyConfiguration(config_file)


log.basicConfig(level=log.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

log.debug("experiment name: %s", pars.experiment_name)
log.debug("NOTE:  This interactive simulation always sends data to MongoDB instance on localhost")

data.set_experiment_name(pars.experiment_name)
data.set_database_hostname("localhost")
data.set_database_port("27017")
config = data.getMingConfiguration()
ming.configure(**config)

sim_id = uuid.uuid4().urn



log.info("Beginning simulation run: %s", sim_id)


beginCollectingData = cpm.expectedIAQuasiStationarityTimeHaploid(pars.popsize,pars.mutationrate)
log.info("Starting data collection at generation: %s", beginCollectingData)

totalSimulationLength = beginCollectingData + pars.length
log.info("Simulation will sample %s generations after stationarity", pars.length)




data.storeSimulationData(pars.popsize,pars.mutationrate,sim_id,pars.samplesize,pars.replications,pars.numloci,__file__,pars.numalleles,simconfig.MAXALLELES)

initial_distribution = utils.constructUniformAllelicDistribution(pars.numalleles)
log.info("Initial allelic distribution: %s", initial_distribution)

pop = sim.Population(size=pars.popsize, ploidy=1, loci=pars.numloci)
simu = sim.Simulator(pop, rep=pars.replications)

simu.evolve(
	initOps = sim.InitGenotype(freq=initial_distribution),
    preOps = [
        sim.PyOperator(func=utils.logGenerationCount, param=(), step=1000, reps=0),
    ],
	matingScheme = sim.RandomSelection(),
	postOps = [sim.KAlleleMutator(k=simconfig.MAXALLELES, rates=pars.mutationrate, loci=sim.ALL_AVAIL),
        sim.PyOperator(func=data.sampleNumAlleles, param=(pars.samplesize, pars.mutationrate, pars.popsize,sim_id,pars.numloci), step=pars.stepsize,begin=beginCollectingData),
        sim.PyOperator(func=data.sampleTraitCounts, param=(pars.samplesize, pars.mutationrate, pars.popsize,sim_id,pars.numloci), step=pars.stepsize,begin=beginCollectingData),
        sim.PyOperator(func=data.censusTraitCounts, param=(pars.mutationrate, pars.popsize,sim_id,pars.numloci), step=pars.stepsize,begin=beginCollectingData),
        sim.PyOperator(func=data.censusNumAlleles, param=(pars.mutationrate, pars.popsize,sim_id,pars.numloci), step=pars.stepsize,begin=beginCollectingData),
        sim.PyOperator(func=data.sampleIndividuals, param=(pars.samplesize, pars.mutationrate, pars.popsize, sim_id,pars.numloci), step=pars.stepsize, begin=beginCollectingData),
		],	
	gen = totalSimulationLength,
)

log.info("Ending simulation run at generation %s", simu.population(0).dvars().gen)



