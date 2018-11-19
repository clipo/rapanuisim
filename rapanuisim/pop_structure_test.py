
import simuPOP
from simuPOP import utils
from simuPOP import sampling
from simuPOP import demography

def calcFst(pop):
    'Calculate Fst and Gst for the whole population and a random sample'
    simuPOP.stat(pop, structure=range(5), vars=['F_st', 'G_st'])
    sample = simuPOP.sampling.drawRandomSample(pop, sizes=[500]*pop.numSubPop())
    simuPOP.stat(sample, structure=range(5), vars=['F_st', 'G_st'])
    print ('Gen: %3d Gst: %.6f (all), %.6f (sample) Fst: %.6f (all) %.6f (sample)' \
        % (pop.dvars().gen,
           pop.dvars().G_st, sample.dvars().G_st,
           pop.dvars().F_st, sample.dvars().F_st))
    return True

pop = simuPOP.Population([10000]*5, loci=[1]*5, infoFields='migrate_to')
pop.evolve(
    initOps = [
        simuPOP.InitSex(),
        simuPOP.InitGenotype(freq=[0.5, 0.5], loci=[0, 2]),
        simuPOP.InitGenotype(freq=[0.2, 0.4, 0.4], loci=[1, 3, 4]),
    ],
    matingScheme = simuPOP.RandomMating(),
    postOps = [
        #simuPOP.Migrator(rate=simuPOP.demography.migrIslandRates(0.01, 3)),
        simuPOP.PyOperator(func=calcFst, step=20),
    ],
    gen = 500
)