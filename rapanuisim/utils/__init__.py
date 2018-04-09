# Copyright (c) $today.year.  Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Creative Commons-GNU General Public Llicense 2.0, as "non-commercial/sharealike".  You may use, modify, and distribute this software for non-commercial purposes, and you must distribute any modifications under the same license.  
#
# For detailed license terms, see:
# http://creativecommons.org/licenses/GPL/2.0/

from rapanuisim.utils.simlogging import logGenerationCount
from rapanuisim.utils.allele_distribution import constructUniformAllelicDistribution
from rapanuisim.utils.script_args import ScriptArgs
from rapanuisim.utils.configuration import CTPyConfiguration
from rapanuisim.utils.parallel import get_parallel_cores
import math

__author__ = 'mark'

# Function for testing the partial or total ordering of a list of numbers

def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))

def non_increasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))


def simulation_burnin_time(popsize, innovrate):
    """
    Calculates burnin time, and rounds it to the nearest 1000 generation interval.

    :param popsize:
    :param innovrate:
    :return:
    """
    tmp = (9.2 * popsize) / (innovrate + 1.0) # this is conservative given the original constant is for the diploid process
    return int(math.ceil(tmp / 1000.0)) * 1000