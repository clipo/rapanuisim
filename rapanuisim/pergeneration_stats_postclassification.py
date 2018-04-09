# Copyright (c) $today.year.  Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Creative Commons-GNU General Public Llicense 2.0, as "non-commercial/sharealike".  You may use, modify, and distribute this software for non-commercial purposes, and you must distribute any modifications under the same license.  
#
# For detailed license terms, see:
# http://creativecommons.org/licenses/GPL/2.0/
"""
.. module:: simrun_stats_postclassification
    :platform: Unix, Windows
    :synopsis: Data object for storing a stats about the post-classification individual sample for a single sim run and generation step.

.. moduleauthor:: Mark E. Madsen <mark@madsenlab.org>

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
    return 'pergeneration_stats_postclassification'


def _get_collection_id():
    """
    :return: returns the collection name for this data object
    """
    return rapanuisim.data.generate_collection_id("_samples_postclassification")




def storePerGenerationStatsPostclassification(generation, classification_id, class_type, class_dim,
                                    coarseness, num_classes, replication, ssize,
                                    popsize, mutation, sim_id, moderichness, classrichness,
                                    mode_iqv, mode_entropy, class_iqv, class_entropy, design_space_occupation, class_innovation_interval_times,neutrality_slatkin):
    PerGenerationStatsPostclassification(dict(
        simulation_time=generation,
        classification_id=classification_id,
        classification_type=class_type,
        classification_dim=class_dim,
        classification_coarseness=coarseness,
        classification_num_classes=num_classes,
        replication=replication,
        sample_size=ssize,
        population_size=popsize,
        mutation_rate=mutation,
        simulation_run_id=sim_id,
        mode_richness=moderichness,
        class_richness=classrichness,
        mode_evenness_iqv=mode_iqv,
        mode_evenness_shannon_entropy=mode_entropy,
        class_evenness_iqv=class_iqv,
        class_shannon_entropy=class_entropy,
        design_space_occupation=design_space_occupation,
        class_innovation_interval_times=class_innovation_interval_times,
        class_neutrality_slatkin=neutrality_slatkin
    )).m.insert()
    return True


def updateFieldPerGenerationStatsPostclassification(record_id, field_name, value):
    record = PerGenerationStatsPostclassification.m.find(dict(_id=record_id)).one()
    record[field_name] = value
    record.m.save()


def columns_to_export_for_analysis():
    cols = [
        "classification_id",
        "classification_type",
        "classification_dim",
        "classification_coarseness",
        "simulation_time",
        "replication",
        "sample_size",
        "population_size",
        "mutation_rate",
        "simulation_run_id",
        "class_richness",
        "class_evenness_iqv",
        "class_shannon_entropy",
        "design_space_occupation",
        "class_neutrality_slatkin",
    ]
    return cols



class PerGenerationStatsPostclassification(Document):

    class __mongometa__:
        session = Session.by_name(_get_dataobj_id())
        name = 'pergeneration_stats_postclassification'
        _id = Field(schema.ObjectId)
        # fields pertaining to classification
        classification_id = Field(str)
        classification_type = Field(str)
        classification_dim = Field(int)
        classification_coarseness = Field(float)
        classification_num_classes = Field(int)

        # fields pertaining to simulation run
        simulation_time = Field(int)
        replication = Field(int)
        sample_size = Field(int)
        population_size = Field(int)
        mutation_rate = Field(float)
        simulation_run_id = Field(str)

        # statistics about this generation of this simulation run
        mode_richness = Field([int])        # a list of richness values, in order of locus ID
        class_richness = Field(int)         # a single value, for the population in this generation of this sim run
        mode_evenness_iqv = Field([float])  # a list of evenness values, in order of locus ID
        mode_evenness_shannon_entropy = Field([float])
        class_evenness_iqv = Field(float)   # a single value, for the population in this generation of this sim run
        class_shannon_entropy = Field(float)
        design_space_occupation = Field(float)  # a single value, denoting the fraction of occupied classes
        class_innovation_interval_times = Field([int])   # a list of intervals between appearances of a new class
        class_neutrality_slatkin = Field(float)