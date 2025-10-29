from evolvekit import *
from evolvekit.operators.Ga.real.mutation.EntropyBasedMutation import (
    EntropyBasedMutation,
)


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
mutation_real = EntropyBasedMutation()
mutation_bin = VirusInfectionMutationBin(
    [
        [0, 1, "*", "*", 0, 1, "*", 0, 1, "*", "*", 0, 1, "*", 0, 1],
        [1, "*", 0, "*", 1, "*", 0, "*", 1, "*", 0, "*", 1, "*", 0, "*"],
    ]
)

island.set_elite_count(20)
island.set_crossover_probability(0.8)
island.set_mutation_probability(0.1)
island.set_max_generations(100)
# island.set_seed(42)
island.set_evaluator(evaluator)
island.set_inspector(inspector)
island.set_operator(selection)
island.set_operator(crossover_real)
# island.set_operator(crossover_bin)
island.set_operator(mutation_real)
# island.set_operator(mutation_bin)
island.set_real_clamp_strategy(GaClampStrategy.RANDOM)
island.set_population_size(100)
results = island.run()

print(f" Total generations: {results.total_generations}")
print(f" Total CPU time: {results.total_time} s")
print(f" Best score: {results.value}")
print(f" Real chromosome of best solution: {results.real_chrom}")
print(f" Binary chromosome of best solution: {results.bin_chrom}")
print()

# (venv) (base) rafal@rafal-Lenovo-ideapad-720-15IKB:~/Documents/studia magisterskie/genesys/EvolveKit$  cd /home/rafal/Documents/studia\ magisterskie/genesys/EvolveKit ; /usr/bin/env /home/rafal/Documents/studia\ magisterskie/genesys/EvolveKit/venv/bin/python /home/rafal/.vscode/extensions/ms-python.debugpy-2025.14.1-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 57553 -- /home/rafal/Documents/studia\ magisterskie/genesys/EvolveKit/mytests/EntropyBasedMutation.py

#  >>> EvolveKit Genetic Algorithm Demo <<<

#  gen: 1, stag: 0, mean: 95094.78, median: 85764.14, stdev: 63690.07, best: 6656.6039
#  gen: 2, stag: 1, mean: 82563.00, median: 83943.28, stdev: 46888.96, best: 6656.6039
#  gen: 3, stag: 2, mean: 96342.11, median: 106836.88, stdev: 50211.49, best: 6656.6039
#  gen: 4, stag: 3, mean: 103003.74, median: 123481.12, stdev: 54957.34, best: 6656.6039
#  gen: 5, stag: 4, mean: 106017.62, median: 123462.45, stdev: 55419.23, best: 6656.6039
#  gen: 6, stag: 5, mean: 115347.75, median: 148618.34, stdev: 55224.12, best: 6656.6039
#  gen: 7, stag: 6, mean: 116832.41, median: 128157.29, stdev: 50837.14, best: 6656.6039
#  gen: 8, stag: 7, mean: 124296.74, median: 155955.35, stdev: 53294.25, best: 6656.6039
#  gen: 9, stag: 8, mean: 118198.02, median: 155955.35, stdev: 57006.86, best: 6656.6039
#  gen: 10, stag: 9, mean: 122796.97, median: 155955.35, stdev: 56186.90, best: 6656.6039
#  gen: 11, stag: 10, mean: 132106.29, median: 160322.61, stdev: 55562.90, best: 6656.6039
#  gen: 12, stag: 0, mean: 108985.45, median: 160322.61, stdev: 63865.81, best: 2638.7686
#  gen: 13, stag: 1, mean: 107493.53, median: 160322.61, stdev: 61882.50, best: 2638.7686
#  gen: 14, stag: 2, mean: 96980.50, median: 136205.66, stdev: 63970.65, best: 2638.7686
#  gen: 15, stag: 3, mean: 117734.60, median: 160322.61, stdev: 61407.27, best: 2638.7686
#  gen: 16, stag: 4, mean: 111207.53, median: 160322.61, stdev: 61309.51, best: 2638.7686
#  gen: 17, stag: 5, mean: 131485.50, median: 160418.43, stdev: 58382.10, best: 2638.7686
#  gen: 18, stag: 6, mean: 119469.33, median: 160418.43, stdev: 62891.57, best: 2638.7686
#  gen: 19, stag: 7, mean: 112090.05, median: 160418.43, stdev: 65068.48, best: 2638.7686
#  gen: 20, stag: 8, mean: 104632.63, median: 155964.15, stdev: 64725.09, best: 2638.7686
#  gen: 21, stag: 9, mean: 106755.50, median: 122382.23, stdev: 61123.81, best: 2638.7686
#  gen: 22, stag: 10, mean: 114786.73, median: 153926.18, stdev: 59119.06, best: 2638.7686
#  gen: 23, stag: 11, mean: 122490.63, median: 160418.43, stdev: 60177.35, best: 2638.7686
#  gen: 24, stag: 12, mean: 125204.33, median: 160418.43, stdev: 56940.97, best: 2638.7686
#  gen: 25, stag: 13, mean: 131295.39, median: 160418.43, stdev: 58332.45, best: 2638.7686
#  gen: 26, stag: 14, mean: 131294.07, median: 160418.43, stdev: 58331.80, best: 2638.7686
#  gen: 27, stag: 15, mean: 131284.86, median: 160418.43, stdev: 58327.18, best: 2638.7686
#  gen: 28, stag: 16, mean: 131284.86, median: 160418.43, stdev: 58327.18, best: 2638.7686
#  gen: 29, stag: 17, mean: 124081.64, median: 160418.43, stdev: 61626.98, best: 2638.7686
#  gen: 30, stag: 18, mean: 131284.86, median: 160418.43, stdev: 58327.18, best: 2638.7686
#  gen: 31, stag: 19, mean: 131284.86, median: 160418.43, stdev: 58327.18, best: 2638.7686
#  gen: 32, stag: 20, mean: 131284.86, median: 160418.43, stdev: 58327.18, best: 2638.7686
#  gen: 33, stag: 21, mean: 131284.86, median: 160418.43, stdev: 58327.18, best: 2638.7686
#  gen: 34, stag: 22, mean: 131284.86, median: 160418.43, stdev: 58327.18, best: 2638.7686
#  gen: 35, stag: 23, mean: 122247.21, median: 160418.43, stdev: 60645.52, best: 2638.7686
#  gen: 36, stag: 24, mean: 86219.03, median: 76935.87, stdev: 64621.42, best: 2638.7686
#  gen: 37, stag: 25, mean: 110200.81, median: 127127.79, stdev: 59893.76, best: 2638.7686
#  gen: 38, stag: 26, mean: 112667.98, median: 155964.15, stdev: 60968.58, best: 2638.7686
#  gen: 39, stag: 27, mean: 117985.27, median: 160418.43, stdev: 62930.46, best: 2638.7686
#  gen: 40, stag: 28, mean: 121605.10, median: 160418.43, stdev: 62981.13, best: 2638.7686
#  gen: 41, stag: 0, mean: 100416.14, median: 141628.29, stdev: 66418.28, best: 1895.8250
#  gen: 42, stag: 1, mean: 94562.33, median: 127200.43, stdev: 64309.75, best: 1895.8250
#  gen: 43, stag: 2, mean: 123316.72, median: 160418.43, stdev: 59043.54, best: 1895.8250
#  gen: 44, stag: 3, mean: 122977.23, median: 160418.43, stdev: 59205.31, best: 1895.8250
#  gen: 45, stag: 4, mean: 126166.03, median: 160418.43, stdev: 60146.38, best: 1895.8250
#  gen: 46, stag: 5, mean: 129956.73, median: 160418.43, stdev: 60932.10, best: 1895.8250
#  gen: 47, stag: 6, mean: 129956.73, median: 160418.43, stdev: 60932.10, best: 1895.8250
#  gen: 48, stag: 7, mean: 129956.73, median: 160418.43, stdev: 60932.10, best: 1895.8250
#  gen: 49, stag: 8, mean: 129956.73, median: 160418.43, stdev: 60932.10, best: 1895.8250
#  gen: 50, stag: 9, mean: 121257.54, median: 160418.43, stdev: 63674.79, best: 1895.8250
#  gen: 51, stag: 10, mean: 117787.47, median: 160418.43, stdev: 64695.96, best: 1895.8250
#  gen: 52, stag: 11, mean: 94416.76, median: 127127.79, stdev: 65757.78, best: 1895.8250
#  gen: 53, stag: 12, mean: 99635.91, median: 134625.68, stdev: 62254.69, best: 1895.8250
#  gen: 54, stag: 13, mean: 116432.48, median: 157470.69, stdev: 62321.36, best: 1895.8250
#  gen: 55, stag: 14, mean: 128828.35, median: 160418.43, stdev: 60761.32, best: 1895.8250
#  gen: 56, stag: 15, mean: 129900.12, median: 160418.43, stdev: 61043.62, best: 1895.8250
#  gen: 57, stag: 16, mean: 129900.12, median: 160418.43, stdev: 61043.62, best: 1895.8250
#  gen: 58, stag: 17, mean: 129900.12, median: 160418.43, stdev: 61043.62, best: 1895.8250
#  gen: 59, stag: 18, mean: 123020.67, median: 160418.43, stdev: 63422.27, best: 1895.8250
#  gen: 60, stag: 19, mean: 129898.67, median: 160418.43, stdev: 61046.49, best: 1895.8250
#  gen: 61, stag: 20, mean: 116274.52, median: 160418.43, stdev: 67796.81, best: 1895.8250
#  gen: 62, stag: 21, mean: 113459.93, median: 160418.43, stdev: 65946.85, best: 1895.8250
#  gen: 63, stag: 22, mean: 129898.67, median: 160418.43, stdev: 61046.49, best: 1895.8250
#  gen: 64, stag: 23, mean: 129898.67, median: 160418.43, stdev: 61046.49, best: 1895.8250
#  gen: 65, stag: 24, mean: 129898.67, median: 160418.43, stdev: 61046.49, best: 1895.8250
#  gen: 66, stag: 25, mean: 129898.67, median: 160418.43, stdev: 61046.49, best: 1895.8250
#  gen: 67, stag: 26, mean: 119449.62, median: 160418.43, stdev: 65331.56, best: 1895.8250
#  gen: 68, stag: 27, mean: 129898.67, median: 160418.43, stdev: 61046.49, best: 1895.8250
#  gen: 69, stag: 28, mean: 129898.67, median: 160418.43, stdev: 61046.49, best: 1895.8250
#  gen: 70, stag: 29, mean: 129898.67, median: 160418.43, stdev: 61046.49, best: 1895.8250
#  gen: 71, stag: 0, mean: 117567.35, median: 160418.43, stdev: 66220.42, best: 702.7582
#  gen: 72, stag: 1, mean: 128556.68, median: 160418.43, stdev: 61000.67, best: 702.7582
#  gen: 73, stag: 2, mean: 128902.68, median: 160418.43, stdev: 61147.75, best: 702.7582
#  gen: 74, stag: 3, mean: 129687.83, median: 160418.43, stdev: 61474.48, best: 702.7582
#  gen: 75, stag: 4, mean: 129687.83, median: 160418.43, stdev: 61474.48, best: 702.7582
#  gen: 76, stag: 5, mean: 129687.83, median: 160418.43, stdev: 61474.48, best: 702.7582
#  gen: 77, stag: 6, mean: 115914.90, median: 160418.43, stdev: 67099.87, best: 702.7582
#  gen: 78, stag: 7, mean: 114165.93, median: 160418.43, stdev: 65783.48, best: 702.7582
#  gen: 79, stag: 8, mean: 102487.87, median: 160418.43, stdev: 68224.25, best: 702.7582
#  gen: 80, stag: 9, mean: 117913.26, median: 160418.43, stdev: 65067.82, best: 702.7582
#  gen: 81, stag: 10, mean: 119784.95, median: 160418.43, stdev: 64768.77, best: 702.7582
#  gen: 82, stag: 11, mean: 129687.83, median: 160418.43, stdev: 61474.48, best: 702.7582
#  gen: 83, stag: 12, mean: 129687.83, median: 160418.43, stdev: 61474.48, best: 702.7582
#  gen: 84, stag: 13, mean: 129687.83, median: 160418.43, stdev: 61474.48, best: 702.7582
#  gen: 85, stag: 14, mean: 129687.83, median: 160418.43, stdev: 61474.48, best: 702.7582
#  gen: 86, stag: 15, mean: 129687.83, median: 160418.43, stdev: 61474.48, best: 702.7582
#  gen: 87, stag: 16, mean: 129687.83, median: 160418.43, stdev: 61474.48, best: 702.7582
#  gen: 88, stag: 17, mean: 129687.83, median: 160418.43, stdev: 61474.48, best: 702.7582
#  gen: 89, stag: 18, mean: 129687.83, median: 160418.43, stdev: 61474.48, best: 702.7582
#  gen: 90, stag: 19, mean: 129687.83, median: 160418.43, stdev: 61474.48, best: 702.7582
#  gen: 91, stag: 20, mean: 129687.83, median: 160418.43, stdev: 61474.48, best: 702.7582
#  gen: 92, stag: 21, mean: 113885.95, median: 160418.43, stdev: 66825.60, best: 702.7582
#  gen: 93, stag: 22, mean: 129687.83, median: 160418.43, stdev: 61474.48, best: 702.7582
#  gen: 94, stag: 23, mean: 117996.48, median: 160418.43, stdev: 66693.22, best: 702.7582
#  gen: 95, stag: 24, mean: 121430.57, median: 160418.43, stdev: 63903.61, best: 702.7582
#  gen: 96, stag: 25, mean: 129687.83, median: 160418.43, stdev: 61474.48, best: 702.7582
#  gen: 97, stag: 26, mean: 129687.83, median: 160418.43, stdev: 61474.48, best: 702.7582
#  gen: 98, stag: 27, mean: 129687.83, median: 160418.43, stdev: 61474.48, best: 702.7582
#  gen: 99, stag: 28, mean: 129687.83, median: 160418.43, stdev: 61474.48, best: 702.7582
#  gen: 100, stag: 29, mean: 105495.19, median: 160418.43, stdev: 68707.26, best: 702.7582
#  gen: 101, stag: 30, mean: 87642.71, median: 98835.97, stdev: 68029.42, best: 702.7582

#  Total generations: 101
#  Total CPU time: 1.6342913870000002 s
#  Best score: 702.7581580457525
#  Real chromosome of best solution: [-0.51186496  1.65189366  1.02763376  0.94883183  0.23654799 -0.77563105
#   1.59855884]
#  Binary chromosome of best solution: []
