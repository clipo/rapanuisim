"""
.. module:: richness_sample
    :platform: Unix, Windows
    :synopsis: Data object for storing a sample of trait richness from a population in MongoDB, via the Ming ORM.

.. moduleauthor:: Mark E. Madsen <mark@madsenlab.org>



    Aggregation framework query for calculating the mean richness for combinations of
    population size, mutation rate, and sample size, for comparison with formulas
    derived from Ewens sampling theory.

    db.richness_sample.aggregate(
        { '$project' : {
            'population_size' : 1,
            'richness' : 1,
            'mutation_rate' : 1,
            'sample_size' : 1,
                    'locus' : 1,
        }},
        {
            '$group' : {
                '_id': { population: '$population_size', mutation_rate : '$mutation_rate', sample_size: '$sample_size', locus: '$locus'},
                'mean_richness' : { '$avg' : '$richness'},
            }
        })

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
    return 'richness'

def _get_collection_id():
    """
    :return: returns the collection name for this data object
    """
    return rapanuisim.data.generate_collection_id("_samples_raw")

def sampleNumAlleles(pop, param):
    """Samples allele richness for all loci in a replicant population, and stores the richness of the sample in the database.

        Args:

            pop (Population):  simuPOP population replicate.

            params (list):  list of parameters (sample size, mutation rate, population size, simulation ID)

        Returns:

            Boolean true:  all PyOperators need to return true.

    """
    (ssize, mutation, popsize,sim_id,numloci) = param
    popID = pop.dvars().rep
    gen = pop.dvars().gen
    sample = drawRandomSample(pop, sizes=ssize)
    sim.stat(sample, alleleFreq=sim.ALL_AVAIL)
    for locus in range(numloci):
        numAlleles = len(sample.dvars().alleleFreq[locus].values())
        _storeRichnessSample(popID,ssize,numAlleles,locus,gen,mutation,popsize,sim_id)
    return True


def _storeRichnessSample(popID, ssize, richness, locus, generation,mutation,popsize,sim_id):
    RichnessSample(dict(
        simulation_time=generation,
        replication=popID,
        locus=locus,
        richness=richness,
        sample_size=ssize,
        population_size=popsize,
        mutation_rate=mutation,
        simulation_run_id=sim_id
    )).m.insert()
    return True




class RichnessSample(Document):

    class __mongometa__:
        session = Session.by_name(_get_dataobj_id())
        name = 'richness_sample'

    _id = Field(schema.ObjectId)
    simulation_time = Field(int)
    replication = Field(int)
    locus = Field(int)
    richness = Field(int)
    sample_size = Field(int)
    population_size = Field(int)
    mutation_rate = Field(float)
    simulation_run_id = Field(str)