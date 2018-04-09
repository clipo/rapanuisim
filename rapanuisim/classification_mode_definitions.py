# Copyright (c) $today.year.  Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Creative Commons-GNU General Public Llicense 2.0, as "non-commercial/sharealike".  You may use, modify, and distribute this software for non-commercial purposes, and you must distribute any modifications under the same license.  
#
# For detailed license terms, see:
# http://creativecommons.org/licenses/GPL/2.0/

"""
.. module:: classification_mode_definitions
    :platform: Unix, Windows
    :synopsis: Data object for storing metadata and parameter about a classification in MongoDB, via the Ming ORM.

.. moduleauthor:: Mark E. Madsen <mark@madsenlab.org>

"""
import logging as log
from ming import Session, Field, schema
from ming.declarative import Document
import rapanuisim

MODETYPE_EVEN = "EVEN"
MODETYPE_RANDOM = "RANDOM"


def _get_dataobj_id():
    """
        Returns the short handle used for this data object in Ming configuration
    """
    return 'classification_modes'

def _get_collection_id():
    """
    :return: returns the collection name for this data object
    """
    return rapanuisim.data.generate_collection_id("_configuration")



def storeClassificationModeDefinition(modetype, maxalleles, nummodes, boundary_map):
    """Stores the parameters and metadata for a simulation run in the database.

        Args:

            modetype (str):  EVEN or RANDOM for type of mode boundaries

            maxalleles (int):  max alleles allowed, fixed for classification and a sim run

            nummodes (int):  number of modes in this partition

            partition (list):  a list of dicts which give upper and lower mode boundaries

        Returns:

            Boolean true:  all PyOperators need to return true.

    """
    ClassificationModeDefinitions(dict(
        mode_type=modetype,
        maxalleles=maxalleles,
        num_modes=nummodes,
        boundary_map=boundary_map,
    )).m.insert()
    return True



mode_boundary = dict(lower=float, upper=float)


class ClassificationModeDefinitions(Document):
    """
    A classification dimension is defined by a set of modes which partition the
    space of the dimension.

    """
    class __mongometa__:
        session = Session.by_name(_get_dataobj_id())
        name = 'classification_modes'
        _id = Field(schema.ObjectId)
        mode_type = Field(str)
        maxalleles = Field(int)
        num_modes = Field(int)
        boundary_map = Field([mode_boundary])


