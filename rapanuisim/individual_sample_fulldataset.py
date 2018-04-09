# Copyright (c) $today.year.  Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Creative Commons-GNU General Public Llicense 2.0, as "non-commercial/sharealike".  You may use, modify, and distribute this software for non-commercial purposes, and you must distribute any modifications under the same license.  
#
# For detailed license terms, see:
# http://creativecommons.org/licenses/GPL/2.0/
"""
.. module:: individual_sample_fulldataset
    :platform: Unix, Windows
    :synopsis: Data object for storing a sample of N individuals (and their genotypes) in MongoDB, via the Ming ORM.

.. moduleauthor:: Mark E. Madsen <mark@madsenlab.org>

This data object and database collection holds individual samples that have been constructed from raw individual_samples
by subsampling, to fill out every desired level of dimensionality and sample size.

"""

import logging as log
from ming import Session, Field, schema
from ming.declarative import Document
import simuPOP as sim
from simuPOP.sampling import drawRandomSample
import pprint as pp
import rapanuisim

def _get_dataobj_id():
    """
        Returns the short handle used for this data object in Ming configuration
    """
    return 'individuals_subsampled'

def _get_collection_id():
    """
    :return: returns the collection name for this data object
    """
    return rapanuisim.data.generate_collection_id("_samples_raw")





def storeIndividualSampleFullDataset(popID, dim, ssize, generation, mutation, popsize, sim_id, sample):
    IndividualSampleFullDataset(dict(
        simulation_time=generation,
        replication=popID,
        dimensionality=dim,
        sample_size=ssize,
        population_size=popsize,
        mutation_rate=mutation,
        simulation_run_id=sim_id,
        sample=sample
    )).m.insert()
    return True




class IndividualSampleFullDataset(Document):

    class __mongometa__:
        session = Session.by_name(_get_dataobj_id())
        name = 'individual_samples_fulldataset'
        _id = Field(schema.ObjectId)
        simulation_time = Field(int)
        replication = Field(int)
        dimensionality=Field(int)
        sample_size = Field(int)
        population_size = Field(int)
        mutation_rate = Field(float)
        simulation_run_id = Field(str)
        # a sample is a list of dicts, where each dict has an individual ID and a list of ints as a value
        sample = Field([
            dict(id=int, genotype=[int])
        ])