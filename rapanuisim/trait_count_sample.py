"""
.. module:: trait_count_sample
    :platform: Unix, Windows
    :synopsis: Data object for storing a sample of trait counts (across all loci) in MongoDB, via the Ming ORM.

.. moduleauthor:: Mark E. Madsen <mark@madsenlab.org>

"""
import logging as log
from ming import Session, Field, schema
from ming.declarative import Document
import simuPOP as sim
from simuPOP.sampling import drawRandomSample
import rapanuisim


def _get_dataobj_id():
    """
        Returns the short handle used for this data object in Ming configuration
    """
    return 'traitcounts'

def _get_collection_id():
    """
    :return: returns the collection name for this data object
    """
    return rapanuisim.data.generate_collection_id("_samples_raw")


def sampleTraitCounts(pop, param):
    """Samples trait counts for all loci in a replicant population, and stores the counts  in the database.

        Args:

            pop (Population):  simuPOP population replicate.

            params (list):  list of parameters (sample size, mutation rate, population size, simulation ID, number of loci)

        Returns:

            Boolean true:  all PyOperators need to return true.

    """
    (ssize, mutation, popsize, sim_id,numloci) = param
    popID = pop.dvars().rep
    gen = pop.dvars().gen
    sample = drawRandomSample(pop, sizes=ssize)
    sim.stat(sample, alleleFreq=sim.ALL_AVAIL)
    for locus in range(numloci):
        alleleMap = sample.dvars().alleleNum[locus]
        for allele,count in alleleMap.iteritems():
            _storeTraitCountSample(popID, ssize, locus, gen, mutation, popsize, sim_id, allele, count)
    return True


def _storeTraitCountSample(popID, ssize, locus, generation, mutation, popsize, sim_id, allele, count):
    TraitCountSample(dict(
        simulation_time=generation,
        replication=popID,
        locus=locus,
        sample_size=ssize,
        population_size=popsize,
        mutation_rate=mutation,
        simulation_run_id=sim_id,
        allele=allele,
        count=count
    )).m.insert()
    return True



class TraitCountSample(Document):

    class __mongometa__:
        session = Session.by_name(_get_dataobj_id())
        name = 'trait_count_sample'

    _id = Field(schema.ObjectId)
    simulation_time = Field(int)
    replication = Field(int)
    locus = Field(int)
    sample_size = Field(int)
    population_size = Field(int)
    mutation_rate = Field(float)
    simulation_run_id = Field(str)
    allele = Field(int)
    count = Field(float)
