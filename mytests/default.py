from evolvekit import *


class Rosenbrock(GaEvaluator):
    def evaluate(self, args: GaEvaluatorArgs) -> float:
        x = args.real_chrom
        return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))

    def extremum(self) -> GaExtremum:
        return GaExtremum.MINIMUM

    def real_domain(self) -> list[tuple[float, float]]:
        return [(-6 + i / 2, 4 + i / 2) for i in range(7)]


class Inspector(GaInspector):
    def inspect(self, stats: GaStatistics) -> GaAction:
        print(f" gen: {stats.generation}, stag: {stats.stagnation}, mean: {stats.mean:.2f}, median: {stats.median:.2f}, stdev: {stats.stdev:.2f}, best: {stats.best_indiv.value:.4f}")
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
mutation_real = VirusInfectionMutation([
    [None, -2.13, None, 0.84, -1.77, None, 5.92],
    [-5.89, None, -2.41, None, None, 2.67, None],
    [None, None, 4.38, None, -3.56, None, None],
    [-2.74, -4.83, None, None, None, 1.45, 6.88],
    [None, None, None, 2.17, -0.94, None, 3.21],
    [None, -1.12, -4.97, None, None, 0.38, None],
    [None, None, None, None, 5.66, None, None],
    [-3.58, None, None, 1.89, None, None, 6.31],
    [None, 3.74, None, -3.92, None, 5.12, None],
    [None, None, None, None, None, None, None],
    [None, -5.23, 1.06, None, None, None, 4.77],
    [None, None, None, None, -2.65, 3.94, None],
    [None, None, -0.58, None, None, None, 6.45],
    [-4.92, None, None, None, 2.88, None, None],
    [None, None, None, 4.51, None, None, 5.03],
    [None, 2.17, None, None, None, -1.36, None],
    [None, None, None, None, None, None, None],
    [None, None, 3.99, None, -1.84, None, None],
    [None, -3.67, None, None, None, None, 6.72],
    [0.12, None, None, -2.43, None, 4.26, None]])
mutation_bin = VirusInfectionMutationBin(
[
    [0, 1, '*', '*', 0, 1, '*', 0, 1, '*', '*', 0, 1, '*', 0, 1],
    [1, '*', 0, '*', 1, '*', 0, '*', 1, '*', 0, '*', 1, '*', 0, '*']
])

island.set_elite_count(2)
island.set_crossover_probability(0.8)
island.set_mutation_probability(0.1)
island.set_max_generations(20)
#island.set_seed(42)
island.set_evaluator(evaluator)
island.set_inspector(inspector)
island.set_operator(selection)
island.set_operator(crossover_real)
#island.set_operator(crossover_bin)
island.set_operator(mutation_real)
#island.set_operator(mutation_bin)
island.set_real_clamp_strategy(GaClampStrategy.RANDOM)
island.set_population_size(20)
results = island.run()

print(f" Total generations: {results.total_generations}")
print(f" Total CPU time: {results.total_time} s")
print(f" Best score: {results.value}")
print(f" Real chromosome of best solution: {results.real_chrom}")
print(f" Binary chromosome of best solution: {results.bin_chrom}")
print()
