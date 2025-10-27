from evolvekit import *
from evolvekit.operators.Ga.real.mutation.SimulatedAnnealingBasedMuttaion1 import (
    SimulatedAnnealingBasedMuttaion1,
)
from evolvekit.operators.Ga.real.mutation.UniformMutation import UniformMutation


class Rosenbrock(GaEvaluator):
    def evaluate(self, args: GaEvaluatorArgs) -> float:
        x = args.real_chrom
        return sum(
            100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
            for i in range(len(x) - 1)
        )

    def extremum(self) -> GaExtremum:
        return GaExtremum.MINIMUM

    def real_domain(self) -> list[tuple[float, float]]:
        return [(-6 + i / 2, 4 + i / 2) for i in range(7)]


class Inspector(GaInspector):
    def inspect(self, stats: GaStatistics) -> GaAction:
        print(
            f" gen: {stats.generation}, stag: {stats.stagnation}, mean: {stats.mean:.2f}, median: {stats.median:.2f}, stdev: {stats.stdev:.2f}, best: {stats.best_indiv.value:.4f}"
        )
        return GaAction.CONTINUE

    def initialize(self):
        print()
        print(" >>> EvolveKit Genetic Algorithm Demo <<<")
        print()

    def finish(self, stats: GaStatistics):
        print()


island = GaIsland()
evaluator = Rosenbrock()
inspector = Inspector()
selection = RankSelection(10)
crossover_real = OnePointCrossover()
crossover_bin = OnePointCrossoverBin()
mutation_real = SimulatedAnnealingBasedMuttaion1()
mutation_bin = VirusInfectionMutationBin(
    [
        [0, 1, "*", "*", 0, 1, "*", 0, 1, "*", "*", 0, 1, "*", 0, 1],
        [1, "*", 0, "*", 1, "*", 0, "*", 1, "*", 0, "*", 1, "*", 0, "*"],
    ]
)

island.set_elite_count(2)
island.set_crossover_probability(0.8)
island.set_mutation_probability(0.1)
island.set_max_generations(20)
# island.set_seed(42)
island.set_evaluator(evaluator)
island.set_inspector(inspector)
island.set_operator(selection)
island.set_operator(crossover_real)
# island.set_operator(crossover_bin)
island.set_operator(mutation_real)
# island.set_operator(mutation_bin)
island.set_real_clamp_strategy(GaClampStrategy.RANDOM)
island.set_population_size(20)
results = island.run()

print(f" Total generations: {results.total_generations}")
print(f" Total CPU time: {results.total_time} s")
print(f" Best score: {results.value}")
print(f" Real chromosome of best solution: {results.real_chrom}")
print(f" Binary chromosome of best solution: {results.bin_chrom}")
print()


#  >>> EvolveKit Genetic Algorithm Demo <<<

#  gen: 1, stag: 0, mean: 89454.64, median: 62081.59, stdev: 71738.87, best: 6823.2738
#  gen: 2, stag: 1, mean: 123562.16, median: 81075.56, stdev: 106988.34, best: 6823.2738
#  gen: 3, stag: 2, mean: 130193.26, median: 114518.65, stdev: 81193.77, best: 6823.2738
#  gen: 4, stag: 3, mean: 169032.24, median: 182110.28, stdev: 93587.69, best: 6823.2738
#  gen: 5, stag: 4, mean: 187101.72, median: 182110.28, stdev: 84201.62, best: 6823.2738
#  gen: 6, stag: 5, mean: 187330.85, median: 182968.72, stdev: 82115.48, best: 6823.2738
#  gen: 7, stag: 6, mean: 187304.72, median: 205221.54, stdev: 86101.55, best: 6823.2738
#  gen: 8, stag: 7, mean: 216127.21, median: 238919.26, stdev: 77950.93, best: 6823.2738
#  gen: 9, stag: 8, mean: 228323.04, median: 274273.66, stdev: 83256.21, best: 6823.2738
#  gen: 10, stag: 0, mean: 222509.58, median: 260872.44, stdev: 96689.69, best: 2372.4863
#  gen: 11, stag: 1, mean: 185688.65, median: 274273.66, stdev: 119375.58, best: 2372.4863
#  gen: 12, stag: 2, mean: 179333.85, median: 229536.26, stdev: 109192.69, best: 2372.4863
#  gen: 13, stag: 3, mean: 183295.43, median: 199684.74, stdev: 89420.43, best: 2372.4863
#  gen: 14, stag: 4, mean: 197907.71, median: 199883.14, stdev: 82661.67, best: 2372.4863
#  gen: 15, stag: 5, mean: 249137.77, median: 274273.66, stdev: 81713.17, best: 2372.4863
#  gen: 16, stag: 6, mean: 211085.21, median: 274273.66, stdev: 102453.32, best: 2372.4863
#  gen: 17, stag: 7, mean: 225474.01, median: 274273.66, stdev: 98664.87, best: 2372.4863
#  gen: 18, stag: 8, mean: 183392.28, median: 224154.19, stdev: 102074.58, best: 2372.4863
#  gen: 19, stag: 9, mean: 231142.60, median: 274273.66, stdev: 89583.99, best: 2372.4863
#  gen: 20, stag: 10, mean: 239224.34, median: 274273.66, stdev: 85737.00, best: 2372.4863
#  gen: 21, stag: 11, mean: 221312.92, median: 274273.66, stdev: 98216.20, best: 2372.4863

#  Total generations: 21
#  Total CPU time: 0.10989672800000005 s
#  Best score: 2372.4862612958277
#  Real chromosome of best solution: [-0.68394182  0.99372731  0.00692099 -1.66442285 -1.13852396 -0.36579477
#  -1.09149996]
#  Binary chromosome of best solution: []
