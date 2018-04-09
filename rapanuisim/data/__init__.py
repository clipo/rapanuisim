# Copyright (c) $today.year.  Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Creative Commons-GNU General Public Llicense 2.0, as "non-commercial/sharealike".  You may use, modify, and distribute this software for non-commercial purposes, and you must distribute any modifications under the same license.  
#
# For detailed license terms, see:
# http://creativecommons.org/licenses/GPL/2.0/

import logging as log

from rapanuisim.data.richness_sample import sampleNumAlleles
from rapanuisim.data.trait_count_sample import sampleTraitCounts
from rapanuisim.data.simulation_data import storeSimulationData, SimulationRun
from rapanuisim.data.trait_lifetime import TraitLifetimeCacheIAModels
from rapanuisim.data.trait_count_population import censusTraitCounts
from rapanuisim.data.classification_data import storeClassificationData, ClassificationData
from rapanuisim.data.classification_mode_definitions import storeClassificationModeDefinition, ClassificationModeDefinitions
from rapanuisim.data.individual_sample_fulldataset import storeIndividualSampleFullDataset, IndividualSampleFullDataset
from rapanuisim.data.richness_population import censusNumAlleles
from rapanuisim.data.experiment_tracking import ExperimentTracking
from rapanuisim.data.pergeneration_stats_postclassification import PerGenerationStatsPostclassification
from rapanuisim.data.pergeneration_stats_traits import PerGenerationStatsTraits
from rapanuisim.data.individual_sample_classified import IndividualSampleClassified
from rapanuisim.data.individual_sample_fulldataset import IndividualSampleFullDataset
from rapanuisim.data.individual_sample import IndividualSample
from rapanuisim.data.persimrun_stats_postclassification import PerSimrunStatsPostclassification

experiment_name = "test"
# the following *should* be overridden by command line processing, even by defaults.
# bogus values are to ensure that CLI processing and configuration is working without bugs
dbhost = "override"
dbport = "override"

# When a new module is added for data, the module's filename should be added to the module list below,

# and the module must support a _get_dataobj_id() method which returns the string used in the Ming ORM
# configuration for MongoDB.  This is defined in each module, and then used in the declarative definition
# of the data object being stored.  Ming configuration is then automatic so that simulation simulations need
# include only two lines which are fully generic.

modules = [ individual_sample, trait_count_population, trait_count_sample,
           richness_sample, richness_population,
           simulation_data, trait_lifetime, classification_data,
           classification_mode_definitions,
           individual_sample_classified,
          individual_sample_fulldataset,
           experiment_tracking,
           pergeneration_stats_postclassification,
           persimrun_stats_postclassification,
           pergeneration_stats_traits]


def getMingConfiguration():
    config = {}
    for module in modules:
        urlstring = 'mongodb://'
        urlstring += dbhost
        urlstring += ":"
        urlstring += dbport
        urlstring += "/"

        key = ''
        key += 'ming.'
        key += module._get_dataobj_id()
        key += '.uri'
        collection = module._get_collection_id()
        urlstring += collection
        #log.debug("Configuring %s module as %s", module, urlstring)
        config[key] = urlstring
    return config


def set_database_hostname(name):
    global dbhost
    dbhost = name

def set_database_port(port):
    global dbport
    dbport = port

def set_experiment_name(name):
    """
    Takes the name of the experiment currently being run, for use as a prefix to database collection names

    :param name:
    :return: none
    """
    global experiment_name
    experiment_name = name


def generate_collection_id(suffix):
    collection_id = experiment_name
    collection_id += suffix
    return collection_id



###### Definitions for reuse aross Ming class boundaries

mode_boundary = dict(lower=float, upper=float)
