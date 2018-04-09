# Copyright (c) $today.year.  Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Creative Commons-GNU General Public Llicense 2.0, as "non-commercial/sharealike".  You may use, modify, and distribute this software for non-commercial purposes, and you must distribute any modifications under the same license.  
#
# For detailed license terms, see:
# http://creativecommons.org/licenses/GPL/2.0/

import logging as log
import simuPOP as sim


class KAlleleLifetimeTrackingMutator(sim.PyOperator):
    """

    """
    def __init__(self, rates, k=100000, loci=sim.ALL_AVAIL, *args, **kwargs):
        sim.PyOperator.__init__(self, func=self.mutate, *args, **kwargs)
        self.k = k
        self.rates = rates
        self.loci = loci



    def mutate(self, pop):
        gen = pop.dvars().gen
        new_allele = sim.kAlleleMutate(pop,self.k, self.rates, self.loci)
        #log.debug("new allele: %s in gen: %s", new_allele, gen)

        # cache the new mutation and the generation
        return True
