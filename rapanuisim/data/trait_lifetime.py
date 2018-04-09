# Copyright (c) $today.year.  Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Creative Commons-GNU General Public Llicense 2.0, as "non-commercial/sharealike".  You may use, modify, and distribute this software for non-commercial purposes, and you must distribute any modifications under the same license.  
#
# For detailed license terms, see:
# http://creativecommons.org/licenses/GPL/2.0/
"""
.. module:: trait_lifetime
    :platform: Unix, Windows
    :synopsis: Classes and data objects for tracking the lifetime of alleles/traits, and storing this in the database.

.. moduleauthor:: Mark E. Madsen <mark@madsenlab.org>

Trait (allele) lifetimes

"""

import logging as log
from ming import Session, Field, schema
from ming.declarative import Document
import simuPOP as sim
import pprint as pp
from collections import defaultdict
import rapanuisim.data


def _get_dataobj_id():
    """
        Returns the short handle used for this data object in Ming configuration
    """
    return 'traitlifetime'


def _get_collection_id():
    """
    :return: returns the collection name for this data object
    """
    return rapanuisim.data.generate_collection_id("_samples_raw")




class TraitLifetimeCacheIAModels:
    """
    Caches the starting generation of each newly mutated allele.  Constructor takes as
     argument the number of loci and initial alleles, since these begin their lifetime at gen 0.

    NOTE:  This class is appropriate ONLY for infinite alleles models, with no
        back-mutation or other mechanisms by which a trait can "come back" from
        an exit.

    """
    def __init__(self, numreps, numloci, numalleles):
        self.origin_cache = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for rep in range(numreps):
            for locus in range(numloci):
                for n in range(numalleles):
                    self.origin_cache[rep][locus][n] = 0


    def _cacheNewAllele(self, rep, allele, locus, gen):
        """Given a simuPOP population, checks for newly mutated alleles and records
            the generation of their first occurrence.  Intended to be used as a
            PyOperator in postOps for evolve()
        """

        self.origin_cache[rep][locus][allele] = gen
        return True


    def debugPrintOriginCache(self):
        pp.pprint(self.origin_cache)


    def allele_origin_tracker(self,mutants):
        """
        Used as an output function in a simuPOP mutator, post 1.11
        """
        for line in mutants.split('\n'):
            if not line:  # handles trailing \n case
                continue
            (gen,loc,ploidy,a1,a2) = line.split('\t')
            #log.debug("cacheNewAllele called for locus %s:  %s -> %s in gen %s", loc, a1, a2, gen)
            self._cacheNewAllele(0,int(a2),int(loc),int(gen))
            #pp.pprint(self.origin_cache)

    def allele_demise_tracker(self, pop, param):
        """Checks to see if any traits have gone to zero frequency and thus
        exited the population.  If a trait has exited, the generation of exit
        is recorded in the cache, and a record inserted into the database
        recording the total lifetime in generations of the trait.

        NOTE:  This operator is appropriate ONLY for infinite alleles models, with no
        back-mutation or other mechanisms by which a trait can "come back" from
        an exit.

        Args:

            pop (Population):  simuPOP population replicate.

            params (list):  list of parameters (sample size, mutation rate, population size, simulation ID, number of loci)

        Returns:

            Boolean true:  all PyOperators need to return true

        """
        (ssize, mutation, popsize, sim_id,numloci) = param
        rep = pop.dvars().rep
        cur_gen = pop.dvars().gen

        sim.stat(pop, alleleFreq=sim.ALL_AVAIL)

        # iterate over loci
        for locus in range(numloci):
            alleles_in_use = self.origin_cache[rep][locus].keys()
            freq = pop.dvars().alleleFreq[locus]

            # zero frequencies do not show up in the sim.stat results, so we infer exit
            # by testing all the alleles in the origin cache for their presence in alleleFreq[locus]
            # any allele that isn't there anymore exited in this step

            for allele in alleles_in_use:
                if allele in freq.keys():
                    #log.debug("allele %s still in population at freq %s", allele, freq[allele])
                    pass
                else:
                    lifetime = self._getLifetimeForExitedAllele(rep,allele,locus,cur_gen)
                    #pp.pprint(freq)
                    #pp.pprint(self.origin_cache)
                    self._storeTraitLifetimeRecord(rep,ssize,mutation,popsize,sim_id,locus,allele,lifetime)

        return True

    def _getLifetimeForExitedAllele(self, rep, allele, locus, gen):
        origin_gen = self.origin_cache[rep][locus][allele]
        lifetime = gen - origin_gen
        #log.debug("gen: %s locus %s: allele %s lifetime: %s", gen, locus, allele, lifetime)
        del self.origin_cache[rep][locus][allele]
        return lifetime

    def _storeTraitLifetimeRecord(self, popID, ssize, mutation, popsize, sim_id, locus, allele, lifetime):
        TraitLifetimeRecord(dict(
            replication=popID,
            locus=locus,
            sample_size=ssize,
            population_size=popsize,
            mutation_rate=mutation,
            simulation_run_id=sim_id,
            allele=allele,
            lifetime=lifetime,
        )).m.insert()
        return True


class TraitLifetimeRecord(Document):

    class __mongometa__:
        session = Session.by_name(_get_dataobj_id())
        name = 'trait_lifetime'

    _id = Field(schema.ObjectId)
    replication = Field(int)
    locus = Field(int)
    population_size = Field(int)
    sample_size = Field(int)
    mutation_rate = Field(float)
    simulation_run_id = Field(str)
    allele = Field(int)
    lifetime = Field(int)










