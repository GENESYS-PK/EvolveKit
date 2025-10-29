from evolvekit import *
from evolvekit.operators.Ga.real.mutation.UniformMutation import UniformMutation
from evolvekit.operators.Ga.real.crossover.FlatCrossover import (
    FlatCrossover,
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
crossover_real = FlatCrossover()
crossover_bin = OnePointCrossoverBin()
mutation_real = UniformMutation()
mutation_bin = VirusInfectionMutationBin(
    [
        [0, 1, "*", "*", 0, 1, "*", 0, 1, "*", "*", 0, 1, "*", 0, 1],
        [1, "*", 0, "*", 1, "*", 0, "*", 1, "*", 0, "*", 1, "*", 0, "*"],
    ]
)

island.set_elite_count(40)
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

#  >>> EvolveKit Genetic Algorithm Demo <<<

#  gen: 1, stag: 0, mean: 95094.78, median: 85764.14, stdev: 63690.07, best: 6656.6039
#  gen: 2, stag: 0, mean: 56540.58, median: 49856.25, stdev: 41923.16, best: 3778.9746
#  gen: 3, stag: 1, mean: 34061.01, median: 29312.72, stdev: 23287.30, best: 3778.9746
#  gen: 4, stag: 0, mean: 17563.68, median: 15172.92, stdev: 11684.05, best: 866.3758
#  gen: 5, stag: 0, mean: 12139.97, median: 9590.44, stdev: 9542.54, best: 456.5792
#  gen: 6, stag: 1, mean: 8649.04, median: 7327.11, stdev: 7076.09, best: 456.5792
#  gen: 7, stag: 2, mean: 6533.54, median: 5366.77, stdev: 3863.86, best: 456.5792
#  gen: 8, stag: 3, mean: 7078.50, median: 5640.69, stdev: 4536.20, best: 456.5792
#  gen: 9, stag: 4, mean: 4302.03, median: 3796.05, stdev: 2855.74, best: 456.5792
#  gen: 10, stag: 5, mean: 3693.35, median: 3252.43, stdev: 1643.52, best: 456.5792
#  gen: 11, stag: 6, mean: 3160.49, median: 2832.95, stdev: 1421.86, best: 456.5792
#  gen: 12, stag: 7, mean: 2640.60, median: 2606.84, stdev: 904.19, best: 456.5792
#  gen: 13, stag: 8, mean: 2410.08, median: 2398.39, stdev: 749.19, best: 456.5792
#  gen: 14, stag: 9, mean: 2342.55, median: 2434.93, stdev: 653.49, best: 456.5792
#  gen: 15, stag: 10, mean: 2566.81, median: 2226.83, stdev: 3187.73, best: 456.5792
#  gen: 16, stag: 11, mean: 2237.23, median: 2149.04, stdev: 786.08, best: 456.5792
#  gen: 17, stag: 12, mean: 2207.22, median: 2135.12, stdev: 772.69, best: 456.5792
#  gen: 18, stag: 13, mean: 2137.45, median: 1862.18, stdev: 835.37, best: 456.5792
#  gen: 19, stag: 14, mean: 2654.48, median: 2345.10, stdev: 5094.56, best: 456.5792
#  gen: 20, stag: 15, mean: 1937.87, median: 1690.87, stdev: 720.22, best: 456.5792
#  gen: 21, stag: 16, mean: 1959.39, median: 1586.32, stdev: 768.98, best: 456.5792
#  gen: 22, stag: 17, mean: 1851.93, median: 1790.54, stdev: 685.58, best: 456.5792
#  gen: 23, stag: 18, mean: 1614.72, median: 1390.22, stdev: 566.17, best: 456.5792
#  gen: 24, stag: 19, mean: 1778.87, median: 1689.42, stdev: 839.51, best: 456.5792
#  gen: 25, stag: 20, mean: 1786.69, median: 2142.04, stdev: 619.98, best: 456.5792
#  gen: 26, stag: 21, mean: 1765.87, median: 1920.07, stdev: 599.17, best: 456.5792
#  gen: 27, stag: 22, mean: 1620.39, median: 1450.80, stdev: 572.09, best: 456.5792
#  gen: 28, stag: 23, mean: 3057.24, median: 1406.53, stdev: 14060.49, best: 456.5792
#  gen: 29, stag: 24, mean: 1428.24, median: 1190.99, stdev: 509.80, best: 456.5792
#  gen: 30, stag: 25, mean: 3236.21, median: 1859.77, stdev: 13222.91, best: 456.5792
#  gen: 31, stag: 26, mean: 1945.63, median: 1731.40, stdev: 1867.81, best: 456.5792
#  gen: 32, stag: 27, mean: 2266.17, median: 1949.93, stdev: 2562.68, best: 456.5792
#  gen: 33, stag: 28, mean: 1569.00, median: 1414.22, stdev: 618.78, best: 456.5792
#  gen: 34, stag: 29, mean: 1543.13, median: 1436.19, stdev: 567.13, best: 456.5792
#  gen: 35, stag: 30, mean: 1391.88, median: 1082.33, stdev: 520.20, best: 456.5792
#  gen: 36, stag: 31, mean: 1553.83, median: 1491.79, stdev: 694.34, best: 456.5792
#  gen: 37, stag: 32, mean: 1540.36, median: 1579.29, stdev: 568.42, best: 456.5792
#  gen: 38, stag: 33, mean: 1365.28, median: 1056.06, stdev: 509.51, best: 456.5792
#  gen: 39, stag: 34, mean: 1394.65, median: 1413.28, stdev: 477.96, best: 456.5792
#  gen: 40, stag: 35, mean: 1415.69, median: 1404.92, stdev: 502.37, best: 456.5792
#  gen: 41, stag: 36, mean: 1281.23, median: 1163.15, stdev: 409.17, best: 456.5792
#  gen: 42, stag: 37, mean: 1266.74, median: 996.90, stdev: 736.15, best: 456.5792
#  gen: 43, stag: 38, mean: 1278.73, median: 1187.82, stdev: 400.17, best: 456.5792
#  gen: 44, stag: 39, mean: 1237.64, median: 975.87, stdev: 428.61, best: 456.5792
#  gen: 45, stag: 40, mean: 1221.23, median: 1184.34, stdev: 391.01, best: 456.5792
#  gen: 46, stag: 41, mean: 1098.25, median: 939.31, stdev: 354.36, best: 456.5792
#  gen: 47, stag: 42, mean: 1166.18, median: 1025.19, stdev: 393.43, best: 456.5792
#  gen: 48, stag: 43, mean: 1565.04, median: 1148.16, stdev: 3788.43, best: 456.5792
#  gen: 49, stag: 44, mean: 1252.87, median: 1396.22, stdev: 437.04, best: 456.5792
#  gen: 50, stag: 45, mean: 1125.32, median: 905.12, stdev: 395.58, best: 456.5792
#  gen: 51, stag: 0, mean: 1152.78, median: 865.88, stdev: 794.75, best: 298.7826
#  gen: 52, stag: 1, mean: 858.04, median: 824.78, stdev: 270.68, best: 298.7826
#  gen: 53, stag: 2, mean: 809.16, median: 783.72, stdev: 194.34, best: 298.7826
#  gen: 54, stag: 3, mean: 813.15, median: 787.12, stdev: 195.10, best: 298.7826
#  gen: 55, stag: 4, mean: 831.35, median: 853.67, stdev: 187.55, best: 298.7826
#  gen: 56, stag: 5, mean: 859.22, median: 919.20, stdev: 205.22, best: 298.7826
#  gen: 57, stag: 6, mean: 1036.29, median: 879.96, stdev: 1709.43, best: 298.7826
#  gen: 58, stag: 7, mean: 831.21, median: 800.54, stdev: 194.12, best: 298.7826
#  gen: 59, stag: 8, mean: 794.55, median: 770.62, stdev: 169.27, best: 298.7826
#  gen: 60, stag: 9, mean: 825.60, median: 846.26, stdev: 223.73, best: 298.7826
#  gen: 61, stag: 10, mean: 821.75, median: 868.46, stdev: 176.08, best: 298.7826
#  gen: 62, stag: 11, mean: 914.82, median: 845.73, stdev: 1110.28, best: 298.7826
#  gen: 63, stag: 12, mean: 841.13, median: 853.46, stdev: 409.15, best: 298.7826
#  gen: 64, stag: 13, mean: 810.51, median: 830.72, stdev: 213.75, best: 298.7826
#  gen: 65, stag: 14, mean: 819.01, median: 908.92, stdev: 203.35, best: 298.7826
#  gen: 66, stag: 15, mean: 756.52, median: 717.97, stdev: 192.12, best: 298.7826
#  gen: 67, stag: 16, mean: 747.27, median: 747.31, stdev: 176.16, best: 298.7826
#  gen: 68, stag: 17, mean: 765.65, median: 681.02, stdev: 566.11, best: 298.7826
#  gen: 69, stag: 18, mean: 748.51, median: 661.90, stdev: 762.50, best: 298.7826
#  gen: 70, stag: 0, mean: 1017.20, median: 625.83, stdev: 1512.29, best: 259.4368
#  gen: 71, stag: 1, mean: 918.93, median: 685.80, stdev: 1016.92, best: 259.4368
#  gen: 72, stag: 2, mean: 646.18, median: 576.65, stdev: 249.10, best: 259.4368
#  gen: 73, stag: 3, mean: 605.31, median: 567.96, stdev: 153.05, best: 259.4368
#  gen: 74, stag: 4, mean: 599.49, median: 561.17, stdev: 156.84, best: 259.4368
#  gen: 75, stag: 0, mean: 575.50, median: 540.29, stdev: 155.01, best: 228.3913
#  gen: 76, stag: 1, mean: 504.70, median: 522.94, stdev: 103.14, best: 228.3913
#  gen: 77, stag: 2, mean: 518.65, median: 496.86, stdev: 260.11, best: 228.3913
#  gen: 78, stag: 3, mean: 507.63, median: 521.20, stdev: 149.97, best: 228.3913
#  gen: 79, stag: 4, mean: 738.75, median: 476.85, stdev: 2512.41, best: 228.3913
#  gen: 80, stag: 0, mean: 450.72, median: 432.69, stdev: 140.62, best: 192.2594
#  gen: 81, stag: 1, mean: 446.83, median: 441.28, stdev: 131.24, best: 192.2594
#  gen: 82, stag: 0, mean: 387.78, median: 326.48, stdev: 151.38, best: 137.1057
#  gen: 83, stag: 0, mean: 1324.93, median: 308.22, stdev: 9612.08, best: 132.9613
#  gen: 84, stag: 1, mean: 753.59, median: 287.28, stdev: 3764.78, best: 132.9613
#  gen: 85, stag: 2, mean: 321.26, median: 284.94, stdev: 104.38, best: 132.9613
#  gen: 86, stag: 3, mean: 298.81, median: 296.92, stdev: 82.47, best: 132.9613
#  gen: 87, stag: 4, mean: 298.89, median: 282.25, stdev: 87.05, best: 132.9613
#  gen: 88, stag: 5, mean: 302.45, median: 248.98, stdev: 104.76, best: 132.9613
#  gen: 89, stag: 6, mean: 324.60, median: 267.62, stdev: 200.24, best: 132.9613
#  gen: 90, stag: 0, mean: 287.28, median: 228.39, stdev: 140.68, best: 68.3720
#  gen: 91, stag: 1, mean: 242.75, median: 218.94, stdev: 87.39, best: 68.3720
#  gen: 92, stag: 2, mean: 224.36, median: 206.15, stdev: 60.98, best: 68.3720
#  gen: 93, stag: 3, mean: 222.81, median: 209.84, stdev: 52.05, best: 68.3720
#  gen: 94, stag: 4, mean: 210.13, median: 208.67, stdev: 36.12, best: 68.3720
#  gen: 95, stag: 5, mean: 215.89, median: 224.57, stdev: 38.92, best: 68.3720
#  gen: 96, stag: 6, mean: 194.46, median: 194.86, stdev: 34.57, best: 68.3720
#  gen: 97, stag: 7, mean: 214.54, median: 197.34, stdev: 223.77, best: 68.3720
#  gen: 98, stag: 8, mean: 204.25, median: 210.50, stdev: 67.89, best: 68.3720
#  gen: 99, stag: 9, mean: 1087.64, median: 210.46, stdev: 8777.38, best: 68.3720
#  gen: 100, stag: 10, mean: 667.38, median: 182.45, stdev: 4400.11, best: 68.3720
#  gen: 101, stag: 11, mean: 251.31, median: 180.56, stdev: 647.54, best: 68.3720

#  Total generations: 101
#  Total CPU time: 2.542431561 s
#  Best score: 68.37196654695161
#  Real chromosome of best solution: [1.22514245 0.84411632 1.02724017 1.06800076 1.40299586 1.84337097
#  3.64100463]
#  Binary chromosome of best solution: []
