# Copyright (c) $today.year.  Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Creative Commons-GNU General Public Llicense 2.0, as "non-commercial/sharealike".  You may use, modify, and distribute this software for non-commercial purposes, and you must distribute any modifications under the same license.  
#
# For detailed license terms, see:
# http://creativecommons.org/licenses/GPL/2.0/
"""
.. module:: allele_distribution
    :platform: Unix, Windows
    :synopsis: Module for creating various initial distributions of traits/alleles, for initializing a simuPOP Population.

.. moduleauthor:: Mark E. Madsen <mark@madsenlab.org>

"""



def constructUniformAllelicDistribution(numalleles):
    """Constructs a uniform distribution of N alleles in the form of a frequency list.

        Args:

            numalleles (int):  Number of alleles present in the initial population.

        Returns:

            (list):  Array of floats, giving the initial frequency of N alleles.

    """
    divisor = 100.0 / numalleles
    frac = divisor / 100.0
    distribution = [frac] * numalleles
    return distribution
