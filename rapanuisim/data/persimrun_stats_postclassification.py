# Copyright (c) $today.year.  Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Creative Commons-GNU General Public Llicense 2.0, as "non-commercial/sharealike".  You may use, modify, and distribute this software for non-commercial purposes, and you must distribute any modifications under the same license.  
#
# For detailed license terms, see:
# http://creativecommons.org/licenses/GPL/2.0/
"""
.. module:: individual_sample_classified
    :platform: Unix, Windows
    :synopsis: Data object for storing a sample of N individuals (with genotype identified to a class) in MongoDB, via the Ming ORM.

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
    return 'persimrun_stats_postclassification'


def _get_collection_id():
    """
    :return: returns the collection name for this data object
    """
    return rapanuisim.data.generate_collection_id("_samples_postclassification")




def storePerSimrunStatsPostclassification(classification_id, class_type, class_dim,
                                    coarseness, replication, ssize,
                                    popsize, mutation, sim_id,class_time_appeared,innovation_mean,innovation_sd):
    PerSimrunStatsPostclassification(dict(
        classification_id=classification_id,
        classification_type=class_type,
        classification_dim=class_dim,
        classification_coarseness=coarseness,
        replication=replication,
        sample_size=ssize,
        population_size=popsize,
        mutation_rate=mutation,
        simulation_run_id=sim_id,
        class_time_first_appearance=class_time_appeared,
        class_innovation_interval_mean=innovation_mean,
        class_innovation_interval_sd=innovation_sd
    )).m.insert()
    return True

def updateFieldPerSimrunStatsPostclassification(record_id, field_name, value):
    record = PerSimrunStatsPostclassification.m.find(dict(_id=record_id)).one()
    record[field_name] = value
    record.m.save()


def columns_to_export_for_analysis():
    cols = [
        "classification_type",
        "classification_dim",
        "classification_coarseness",
        "simulation_time",
        "replication",
        "sample_size",
        "population_size",
        "mutation_rate",
        "simulation_run_id",
        "class_innovation_interval_mean",
        "class_innovation_interval_sd"
    ]
    return cols



class PerSimrunStatsPostclassification(Document):

    class __mongometa__:
        session = Session.by_name(_get_dataobj_id())
        name = 'persimrun_stats_postclassification'
        _id = Field(schema.ObjectId)

        # pertaining to simulation run
        replication = Field(int)
        sample_size = Field(int)
        population_size = Field(int)
        mutation_rate = Field(float)
        simulation_run_id = Field(str)

        # fields pertaining to classification
        classification_id = Field(str)
        classification_type = Field(str)
        classification_dim = Field(int)
        classification_coarseness = Field(float)


        class_time_first_appearance = Field([dict(classid=str,time=int)])
        class_innovation_interval_mean = Field(float)
        class_innovation_interval_sd = Field(float)