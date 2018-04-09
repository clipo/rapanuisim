# Copyright (c) $today.year.  Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Creative Commons-GNU General Public Llicense 2.0, as "non-commercial/sharealike".  You may use, modify, and distribute this software for non-commercial purposes, and you must distribute any modifications under the same license.  
#
# For detailed license terms, see:
# http://creativecommons.org/licenses/GPL/2.0/

"""Mathematical functions and equations relating to CT models and population genetics models.

    Modules are named for the stochastic process they represent, with methods for
    major functions or equations.

.. moduleauthor:: Mark E. Madsen <mark@madsenlab.org>

"""

from rapanuisim.math.diversity import diversity_shannon_entropy, diversity_iqv, diversity_neiman_tf
from rapanuisim.math.wright_fisher_process import expectedIAQuasiStationarityTimeHaploid
from rapanuisim.math.trait_statistics import TraitStatisticsPerSample