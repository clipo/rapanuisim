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
from datetime import datetime


__author__ = 'mark'

stage_map = {
    "description" : "description",
    "simulation" : "sim_data_collected",
    "subsampling" : "subsampling_complete",
    "classification" : "classification_complete",
    "postclassification-stats" : "postclassification_simrun_stats_complete",
    "timeaveraging" : "timeaveraging_complete",
    "timeaveraging-classified" : "ta_classification_complete",
    "trait-stats" : "trait_statistics_complete",
}



def _get_dataobj_id():
    """
        Returns the short handle used for this data object in Ming configuration
    """
    return 'experiment_tracking'

def _get_collection_id():
    """
    ALWAYS RETURNS A FIXED COLLECTION NAME, SINCE THIS IS A CENTRAL REGISTRY OF CTPY EXPERIMENTS
    :return: returns the collection name for this data object
    """
    return "ctpy_registry"


def storeCompleteExperimentRecord(name,exp_start_time,description,sim_data_complete,sim_data_time,subsampling_complete,subsampling_time,
    classification_complete,classification_time,postclass_stats,postclass_stats_time,timeaveraging_complete,timeaveraging_time,ta_classification_complete,
    ta_classification_time,trait_stat_complete,trait_stat_time,exp_end_proc_time):
    """
    Provides tracking information for an experiment in the database
    The complete record will rarely be used in production, this is mainly for testing and building tools.
    """
    ExperimentTracking(dict(
        experiment_name = name,
        experiment_begin_tstamp = exp_start_time,
        description = description,
        sim_data_collected = sim_data_complete,
        sim_data_tstamp = sim_data_time,
        subsampling_complete = subsampling_complete,
        subsampling_tstamp = subsampling_time,
        classification_complete = classification_complete,
        classification_tstamp = classification_time,
        postclassification_simrun_stats_complete = postclass_stats,
        postclassification_simrun_stats_tstamp = postclass_stats_time,
        timeaveraging_complete = timeaveraging_complete,
        timeaveraging_tstamp = timeaveraging_time,
        ta_classification_complete = ta_classification_complete,
        ta_classification_tstamp = ta_classification_time,
        trait_statistics_complete = trait_stat_complete,
        trait_statistics_tstamp = trait_stat_time,
        experiment_end_processing_time = exp_end_proc_time,
    )).m.insert()
    return True


def initializeExperimentRecord(name,exp_start_time):
    ExperimentTracking(dict(
        experiment_name = name,
        experiment_begin_tstamp = exp_start_time
    )).m.insert()


def get_experiment_stage_tags():
    """
    Get a set of keys that can be used outside the module for selection (e.g., on CLI switches) of
    fields for updating.

    :return: dict of strings
    """
    return stage_map.keys()


def update_field_by_stage_tag(experiment_name, tag_name, new_value):
    db_field = stage_map[tag_name]
    _update_field(experiment_name, db_field, new_value)


def _update_field(record_id, field_name, value):
    record = ExperimentTracking.m.find(dict(experiment_name = record_id)).one()
    record[field_name] = value
    record.m.save()


class ExperimentTracking(Document):

    class __mongometa__:
        session = Session.by_name(_get_dataobj_id())
        name = 'experiment_tracking'

    _id = Field(schema.ObjectId)
    experiment_name = Field(str)
    experiment_begin_tstamp = Field(datetime)
    description = Field(str)
    sim_data_collected = Field(bool)
    sim_data_tstamp = Field(datetime)
    subsampling_complete = Field(bool)
    subsampling_tstamp = Field(datetime)
    classification_complete = Field(bool)
    classification_tstamp = Field(datetime)
    postclassification_simrun_stats_complete = Field(bool)
    postclassification_simrun_stats_tstamp = Field(datetime)
    timeaveraging_complete = Field(bool)
    timeaveraging_tstamp = Field(datetime)
    ta_classification_complete = Field(bool)
    ta_classification_tstamp = Field(datetime)
    trait_statistics_complete = Field(bool)
    trait_statistics_tstamp = Field(datetime)
    experiment_end_processing_time = Field(datetime)

