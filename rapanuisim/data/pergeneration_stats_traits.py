# Copyright (c) $today.year.  Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Creative Commons-GNU General Public Llicense 2.0, as "non-commercial/sharealike".  You may use, modify, and distribute this software for non-commercial purposes, and you must distribute any modifications under the same license.  
#
# For detailed license terms, see:
# http://creativecommons.org/licenses/GPL/2.0/
"""
.. module:: pergeneration_stats_traits
    :platform: Unix, Windows
    :synopsis: Data object for storing stats about raw traits for the individual full dataset samples for a single sim run and generation step.

.. moduleauthor:: Mark E. Madsen <mark@madsenlab.org>

"""

import logging as log
from ming import Session, Field, schema
from ming.declarative import Document
import simuPOP as sim
from simuPOP.sampling import drawRandomSample
import pprint as pp
import rapanuisim.data

def _get_dataobj_id():
    """
        Returns the short handle used for this data object in Ming configuration
    """
    return 'pergeneration_stats_traits'


def _get_collection_id():
    """
    :return: returns the collection name for this data object
    """
    return rapanuisim.data.generate_collection_id("_samples_postclassification")




def storePerGenerationStatsTraits(generation, replication, ssize,
                                    popsize, mutation, dimensionality, sim_id, mean_richness, mean_entropy,
                                    mean_iqv, loci_richness, loci_entropy, loci_iqv,loci_neutrality_slatkin,mean_slatkin):
    PerGenerationStatsTraits(dict(
        simulation_time=generation,
        replication=replication,
        sample_size=ssize,
        population_size=popsize,
        mutation_rate=mutation,
        dimensionality=dimensionality,
        simulation_run_id=sim_id,
        mean_trait_richness = mean_richness,
        mean_evenness_shannon_entropy = mean_entropy,
        mean_evenness_iqv = mean_iqv,
        loci_trait_richness = loci_richness,
        loci_evenness_shannon_entropy = loci_entropy,
        loci_evenness_iqv = loci_iqv,
        loci_neutrality_slatkin = loci_neutrality_slatkin,
        mean_neutrality_slatkin = mean_slatkin,
    )).m.insert()
    return True


def updateFieldPerGenerationStatsTraits(record_id, field_name, value):
    record = PerGenerationStatsTraits.m.find(dict(_id=record_id)).one()
    record[field_name] = value
    record.m.save()


def columns_to_export_for_analysis():
    cols = [
        "simulation_run_id",
        "simulation_time",
        "replication",
        "sample_size",
        "population_size",
        "mutation_rate",
        "dimensionality",
        "mean_trait_richness",
        "mean_evenness_shannon_entropy",
        "mean_evenness_iqv",
        "mean_neutrality_slatkin"
    ]
    return cols



class PerGenerationStatsTraits(Document):

    class __mongometa__:
        session = Session.by_name(_get_dataobj_id())
        name = 'pergeneration_stats_traits'
        _id = Field(schema.ObjectId)

        # fields pertaining to simulation run
        simulation_run_id = Field(str)
        simulation_time = Field(int)
        replication = Field(int)
        sample_size = Field(int)
        population_size = Field(int)
        mutation_rate = Field(float)
        dimensionality = Field(int)


        # statistics about this generation of this simulation run
        mean_trait_richness = Field(float)
        mean_evenness_shannon_entropy = Field(float)
        mean_evenness_iqv = Field(float)
        loci_trait_richness = Field([int])
        loci_evenness_shannon_entropy = Field([float])
        loci_evenness_iqv = Field([float])
        loci_neutrality_slatkin = Field([float])
        mean_neutrality_slatkin = Field(float)