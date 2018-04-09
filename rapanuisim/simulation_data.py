# Copyright (c) $today.year.  Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Creative Commons-GNU General Public Llicense 2.0, as "non-commercial/sharealike".  You may use, modify, and distribute this software for non-commercial purposes, and you must distribute any modifications under the same license.  
#
# For detailed license terms, see:
# http://creativecommons.org/licenses/GPL/2.0/

"""
.. module:: simulation_data
    :platform: Unix, Windows
    :synopsis: Data object for storing metadata and parameter about a simulation run in MongoDB, via the Ming ORM.

.. moduleauthor:: Mark E. Madsen <mark@madsenlab.org>

"""
import logging as log
from ming import Session, Field, schema
from ming.declarative import Document
import simuPOP as sim
from simuPOP.sampling import drawRandomSample
import rapanuisim

__author__ = 'mark'

def _get_dataobj_id():
    """
        Returns the short handle used for this data object in Ming configuration
    """
    return 'simulations'

def _get_collection_id():
    """
    :return: returns the collection name for this data object
    """
    return rapanuisim.data.generate_collection_id("_samples_raw")



def storeSimulationData(popsize,mutation,sim_id,ssize,replicates,num_loci,script,numalleles,maxalleles):
    """Stores the parameters and metadata for a simulation run in the database.

        Args:

            popsize (int):  Population size

            mutation (float):  mutation rate

            sim_id (str):  UUID for this simulation run

            sample size (int):  Number of individuals sampled to calculate various observables (richness, trait frequencies, etc)

            replicates (int):  Number of independent populations to evolve with the same set of parameters

            num_loci (int):  Number of trait dimensions (loci) that individuals are equipped with.

            script (str):  Pathname to the simuPOP simulation script used for this simulation run

            numalleles (int):  Number of initial alleles used to initialize each locus, prior to starting the simulation.

        Returns:

            Boolean true:  all PyOperators need to return true.

    """
    SimulationRun(dict(
        replicates=replicates,
        population_size=popsize,
        mutation_rate=mutation,
        simulation_run_id=sim_id,
        sample_size=ssize,
        num_loci=num_loci,
        script_filename=script,
        num_initial_alleles=numalleles,
        max_alleles=maxalleles
    )).m.insert()
    return True





class SimulationRun(Document):

    class __mongometa__:
        session = Session.by_name(_get_dataobj_id())
        name = 'simulation_runs'

    _id = Field(schema.ObjectId)
    script_filename = Field(str)
    replicates = Field(int)
    num_loci = Field(int)
    sample_size = Field(int)
    population_size = Field(int)
    mutation_rate = Field(float)
    simulation_run_id = Field(str)
    num_initial_alleles = Field(int)
    max_alleles = Field(int)

