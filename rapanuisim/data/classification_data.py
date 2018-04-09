# Copyright (c) $today.year.  Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Creative Commons-GNU General Public Llicense 2.0, as "non-commercial/sharealike".  You may use, modify, and distribute this software for non-commercial purposes, and you must distribute any modifications under the same license.  
#
# For detailed license terms, see:
# http://creativecommons.org/licenses/GPL/2.0/

"""
.. module:: classification_data
    :platform: Unix, Windows
    :synopsis: Data object for storing metadata and parameter about a classification in MongoDB, via the Ming ORM.

.. moduleauthor:: Mark E. Madsen <mark@madsenlab.org>

"""
import logging as log
from ming import Session, Field, schema
from ming.declarative import Document
import rapanuisim.data


__author__ = 'mark'

def _get_dataobj_id():
    """
        Returns the short handle used for this data object in Ming configuration
    """
    return 'classifications'


def _get_collection_id():
    """
    :return: returns the collection name for this data object
    """
    return rapanuisim.data.generate_collection_id("_configuration")


def storeClassificationData(class_type, maxalleles, num_loci, coarseness, dimlist):
    """Stores the parameters and metadata for a simulation run in the database.

        Args:

            classification_type (str):  EVEN or RANDOM for type of mode boundaries

            maxalleles (int):  max alleles allowed, fixed for classification and a sim run

            num_loci (int):  number of dimensions or loci in the classification and sim runs

            dimlist (list): a dicts, with dimension indices as keys, and a reference to a mode as value

        Returns:

            Boolean true:  all PyOperators need to return true.

    """
    ClassificationData(dict(
        classification_type=class_type,
        maxalleles=maxalleles,
        dimensions=num_loci,
        mean_coarseness=coarseness,
        modes_for_dimensions=dimlist,
    )).m.insert()
    return True





class ClassificationData(Document):
    """
    A classification is represented by a dict of dimensions, each of which points to a mode definition

    """
    class __mongometa__:
        session = Session.by_name(_get_dataobj_id())
        name = 'classifications'

    _id = Field(schema.ObjectId)
    classification_type = Field(str)
    maxalleles = Field(int) # a given classification is relative to maxalleles
    dimensions = Field(int) # i.e., numloci
    mean_coarseness = Field(float)
    modes_for_dimensions = Field([schema.ObjectId])


