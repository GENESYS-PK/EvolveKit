from evolvekit import *
from evolvekit.operators.Ga.real.mutation.UniformMutation import UniformMutation
from evolvekit.operators.Ga.real.crossover.BlendCrossoverAlpha import (
    BlendCrossoverAlpha,
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
crossover_real = BlendCrossoverAlpha()
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
island.set_max_generations(300)
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

# >>> EvolveKit Genetic Algorithm Demo <<<

#  gen: 1, stag: 0, mean: 95094.78, median: 85764.14, stdev: 63690.07, best: 6656.6039
#  gen: 2, stag: 1, mean: 57807.87, median: 50685.48, stdev: 38440.35, best: 6656.6039
#  gen: 3, stag: 0, mean: 46816.29, median: 35440.35, stdev: 39509.44, best: 3563.1148
#  gen: 4, stag: 0, mean: 33681.09, median: 25409.00, stdev: 28822.70, best: 983.4044
#  gen: 5, stag: 1, mean: 27164.95, median: 17735.54, stdev: 25156.51, best: 983.4044
#  gen: 6, stag: 2, mean: 32718.12, median: 15105.23, stdev: 32512.44, best: 983.4044
#  gen: 7, stag: 0, mean: 25107.23, median: 12018.93, stdev: 27910.66, best: 284.9359
#  gen: 8, stag: 1, mean: 13233.37, median: 7964.59, stdev: 18058.24, best: 284.9359
#  gen: 9, stag: 2, mean: 11529.20, median: 6591.30, stdev: 17582.60, best: 284.9359
#  gen: 10, stag: 3, mean: 10347.43, median: 4917.11, stdev: 17577.97, best: 284.9359
#  gen: 11, stag: 4, mean: 11990.73, median: 3868.79, stdev: 23781.87, best: 284.9359
#  gen: 12, stag: 5, mean: 10038.23, median: 3465.87, stdev: 15138.05, best: 284.9359
#  gen: 13, stag: 6, mean: 14094.85, median: 3480.18, stdev: 18817.44, best: 284.9359
#  gen: 14, stag: 7, mean: 18095.84, median: 3334.08, stdev: 26305.29, best: 284.9359
#  gen: 15, stag: 8, mean: 25285.08, median: 4969.72, stdev: 34480.20, best: 284.9359
#  gen: 16, stag: 9, mean: 20993.95, median: 5336.75, stdev: 31733.74, best: 284.9359
#  gen: 17, stag: 10, mean: 9627.37, median: 2230.57, stdev: 17187.49, best: 284.9359
#  gen: 18, stag: 11, mean: 8883.31, median: 2411.29, stdev: 13336.81, best: 284.9359
#  gen: 19, stag: 12, mean: 7554.72, median: 1671.33, stdev: 14501.37, best: 284.9359
#  gen: 20, stag: 13, mean: 12413.51, median: 1523.21, stdev: 19200.21, best: 284.9359
#  gen: 21, stag: 14, mean: 11076.94, median: 3264.29, stdev: 16065.40, best: 284.9359
#  gen: 22, stag: 15, mean: 12110.68, median: 2619.09, stdev: 17221.06, best: 284.9359
#  gen: 23, stag: 16, mean: 6949.33, median: 1135.88, stdev: 15141.20, best: 284.9359
#  gen: 24, stag: 17, mean: 1763.69, median: 981.80, stdev: 4209.35, best: 284.9359
#  gen: 25, stag: 0, mean: 1025.48, median: 904.89, stdev: 595.47, best: 268.0527
#  gen: 26, stag: 1, mean: 1088.28, median: 811.33, stdev: 832.51, best: 268.0527
#  gen: 27, stag: 0, mean: 1046.70, median: 711.57, stdev: 809.60, best: 233.0421
#  gen: 28, stag: 1, mean: 775.83, median: 617.83, stdev: 412.81, best: 233.0421
#  gen: 29, stag: 2, mean: 1169.64, median: 962.86, stdev: 2282.92, best: 233.0421
#  gen: 30, stag: 3, mean: 843.16, median: 855.57, stdev: 412.54, best: 233.0421
#  gen: 31, stag: 4, mean: 853.39, median: 734.74, stdev: 744.03, best: 233.0421
#  gen: 32, stag: 0, mean: 698.96, median: 581.58, stdev: 394.92, best: 226.7658
#  gen: 33, stag: 0, mean: 735.67, median: 490.00, stdev: 584.71, best: 215.1403
#  gen: 34, stag: 1, mean: 838.58, median: 713.50, stdev: 577.89, best: 215.1403
#  gen: 35, stag: 2, mean: 888.01, median: 964.64, stdev: 508.60, best: 215.1403
#  gen: 36, stag: 3, mean: 760.79, median: 560.98, stdev: 468.07, best: 215.1403
#  gen: 37, stag: 4, mean: 652.75, median: 545.09, stdev: 406.22, best: 215.1403
#  gen: 38, stag: 5, mean: 646.52, median: 486.39, stdev: 438.52, best: 215.1403
#  gen: 39, stag: 0, mean: 716.34, median: 380.61, stdev: 2197.81, best: 143.4011
#  gen: 40, stag: 1, mean: 424.92, median: 356.32, stdev: 244.25, best: 143.4011
#  gen: 41, stag: 2, mean: 615.35, median: 333.99, stdev: 1939.23, best: 143.4011
#  gen: 42, stag: 3, mean: 372.60, median: 302.55, stdev: 203.75, best: 143.4011
#  gen: 43, stag: 4, mean: 449.79, median: 331.77, stdev: 289.34, best: 143.4011
#  gen: 44, stag: 0, mean: 497.89, median: 396.28, stdev: 313.68, best: 81.3626
#  gen: 45, stag: 1, mean: 1764.82, median: 385.52, stdev: 12591.71, best: 81.3626
#  gen: 46, stag: 2, mean: 329.08, median: 249.21, stdev: 186.93, best: 81.3626
#  gen: 47, stag: 3, mean: 319.38, median: 224.46, stdev: 207.16, best: 81.3626
#  gen: 48, stag: 4, mean: 402.63, median: 227.92, stdev: 506.51, best: 81.3626
#  gen: 49, stag: 5, mean: 412.00, median: 348.73, stdev: 589.51, best: 81.3626
#  gen: 50, stag: 6, mean: 425.95, median: 205.02, stdev: 584.48, best: 81.3626
#  gen: 51, stag: 7, mean: 408.89, median: 320.58, stdev: 393.00, best: 81.3626
#  gen: 52, stag: 8, mean: 455.33, median: 228.83, stdev: 1438.25, best: 81.3626
#  gen: 53, stag: 9, mean: 1218.20, median: 241.74, stdev: 4657.37, best: 81.3626
#  gen: 54, stag: 10, mean: 1805.24, median: 253.42, stdev: 7664.15, best: 81.3626
#  gen: 55, stag: 11, mean: 377.50, median: 218.91, stdev: 630.94, best: 81.3626
#  gen: 56, stag: 12, mean: 407.66, median: 197.60, stdev: 692.78, best: 81.3626
#  gen: 57, stag: 13, mean: 243.93, median: 167.67, stdev: 221.18, best: 81.3626
#  gen: 58, stag: 14, mean: 1284.26, median: 162.98, stdev: 10504.72, best: 81.3626
#  gen: 59, stag: 15, mean: 3892.15, median: 179.90, stdev: 17325.59, best: 81.3626
#  gen: 60, stag: 16, mean: 417.54, median: 192.40, stdev: 1915.41, best: 81.3626
#  gen: 61, stag: 17, mean: 229.31, median: 175.22, stdev: 114.56, best: 81.3626
#  gen: 62, stag: 18, mean: 217.38, median: 144.77, stdev: 133.19, best: 81.3626
#  gen: 63, stag: 19, mean: 226.81, median: 140.58, stdev: 148.95, best: 81.3626
#  gen: 64, stag: 20, mean: 298.92, median: 171.58, stdev: 447.73, best: 81.3626
#  gen: 65, stag: 21, mean: 289.33, median: 226.86, stdev: 224.74, best: 81.3626
#  gen: 66, stag: 22, mean: 315.25, median: 308.75, stdev: 205.49, best: 81.3626
#  gen: 67, stag: 23, mean: 362.08, median: 139.59, stdev: 1006.43, best: 81.3626
#  gen: 68, stag: 24, mean: 277.57, median: 232.16, stdev: 214.96, best: 81.3626
#  gen: 69, stag: 25, mean: 265.17, median: 154.11, stdev: 180.94, best: 81.3626
#  gen: 70, stag: 26, mean: 281.20, median: 141.97, stdev: 209.31, best: 81.3626
#  gen: 71, stag: 27, mean: 299.07, median: 166.63, stdev: 210.11, best: 81.3626
#  gen: 72, stag: 28, mean: 976.50, median: 130.49, stdev: 7537.01, best: 81.3626
#  gen: 73, stag: 29, mean: 284.01, median: 161.29, stdev: 249.84, best: 81.3626
#  gen: 74, stag: 30, mean: 284.21, median: 185.65, stdev: 233.46, best: 81.3626
#  gen: 75, stag: 0, mean: 1772.48, median: 127.97, stdev: 14162.98, best: 81.0654
#  gen: 76, stag: 1, mean: 906.23, median: 138.99, stdev: 6642.52, best: 81.0654
#  gen: 77, stag: 2, mean: 493.10, median: 146.61, stdev: 1262.66, best: 81.0654
#  gen: 78, stag: 3, mean: 171.23, median: 156.92, stdev: 79.24, best: 81.0654
#  gen: 79, stag: 4, mean: 156.96, median: 165.90, stdev: 49.06, best: 81.0654
#  gen: 80, stag: 5, mean: 157.33, median: 163.21, stdev: 57.55, best: 81.0654
#  gen: 81, stag: 6, mean: 146.81, median: 148.16, stdev: 43.24, best: 81.0654
#  gen: 82, stag: 7, mean: 144.29, median: 123.82, stdev: 50.18, best: 81.0654
#  gen: 83, stag: 8, mean: 150.21, median: 137.46, stdev: 53.42, best: 81.0654
#  gen: 84, stag: 9, mean: 137.44, median: 117.08, stdev: 45.14, best: 81.0654
#  gen: 85, stag: 10, mean: 158.90, median: 109.60, stdev: 270.31, best: 81.0654
#  gen: 86, stag: 11, mean: 124.67, median: 106.76, stdev: 49.87, best: 81.0654
#  gen: 87, stag: 12, mean: 122.87, median: 105.64, stdev: 40.05, best: 81.0654
#  gen: 88, stag: 13, mean: 365.69, median: 113.77, stdev: 2373.51, best: 81.0654
#  gen: 89, stag: 14, mean: 2034.50, median: 120.27, stdev: 6686.86, best: 81.0654
#  gen: 90, stag: 15, mean: 1231.75, median: 117.09, stdev: 6237.41, best: 81.0654
#  gen: 91, stag: 16, mean: 663.76, median: 100.02, stdev: 2613.43, best: 81.0654
#  gen: 92, stag: 17, mean: 821.08, median: 96.70, stdev: 3235.53, best: 81.0654
#  gen: 93, stag: 18, mean: 268.65, median: 95.98, stdev: 521.04, best: 81.0654
#  gen: 94, stag: 19, mean: 425.44, median: 97.71, stdev: 886.11, best: 81.0654
#  gen: 95, stag: 0, mean: 381.82, median: 93.27, stdev: 839.67, best: 77.7251
#  gen: 96, stag: 0, mean: 547.68, median: 91.66, stdev: 1372.16, best: 75.4819
#  gen: 97, stag: 1, mean: 586.03, median: 149.58, stdev: 828.00, best: 75.4819
#  gen: 98, stag: 2, mean: 646.34, median: 138.68, stdev: 1070.02, best: 75.4819
#  gen: 99, stag: 3, mean: 882.78, median: 101.72, stdev: 1634.03, best: 75.4819
#  gen: 100, stag: 4, mean: 1143.66, median: 290.24, stdev: 2030.62, best: 75.4819
#  gen: 101, stag: 0, mean: 1184.12, median: 118.53, stdev: 2355.69, best: 74.1695
#  gen: 102, stag: 0, mean: 1073.60, median: 105.33, stdev: 1472.50, best: 70.3740
#  gen: 103, stag: 0, mean: 1388.41, median: 90.47, stdev: 1837.28, best: 67.8976
#  gen: 104, stag: 1, mean: 2330.13, median: 451.50, stdev: 4574.99, best: 67.8976
#  gen: 105, stag: 2, mean: 1014.82, median: 103.66, stdev: 2010.84, best: 67.8976
#  gen: 106, stag: 3, mean: 1951.80, median: 220.52, stdev: 2710.05, best: 67.8976
#  gen: 107, stag: 4, mean: 1485.92, median: 1024.38, stdev: 1818.74, best: 67.8976
#  gen: 108, stag: 5, mean: 1110.19, median: 574.94, stdev: 1419.24, best: 67.8976
#  gen: 109, stag: 6, mean: 855.30, median: 101.28, stdev: 1381.46, best: 67.8976
#  gen: 110, stag: 7, mean: 435.34, median: 82.35, stdev: 929.06, best: 67.8976
#  gen: 111, stag: 8, mean: 696.17, median: 95.65, stdev: 1266.57, best: 67.8976
#  gen: 112, stag: 9, mean: 403.65, median: 83.09, stdev: 986.59, best: 67.8976
#  gen: 113, stag: 10, mean: 686.99, median: 99.92, stdev: 1690.26, best: 67.8976
#  gen: 114, stag: 11, mean: 1904.46, median: 99.87, stdev: 5251.53, best: 67.8976
#  gen: 115, stag: 12, mean: 1466.02, median: 131.71, stdev: 2831.93, best: 67.8976
#  gen: 116, stag: 13, mean: 352.53, median: 80.01, stdev: 516.40, best: 67.8976
#  gen: 117, stag: 14, mean: 410.45, median: 99.11, stdev: 605.12, best: 67.8976
#  gen: 118, stag: 15, mean: 513.86, median: 90.52, stdev: 752.36, best: 67.8976
#  gen: 119, stag: 16, mean: 467.74, median: 85.14, stdev: 754.70, best: 67.8976
#  gen: 120, stag: 17, mean: 442.00, median: 91.90, stdev: 638.64, best: 67.8976
#  gen: 121, stag: 18, mean: 303.00, median: 77.41, stdev: 400.70, best: 67.8976
#  gen: 122, stag: 19, mean: 352.71, median: 165.60, stdev: 397.22, best: 67.8976
#  gen: 123, stag: 20, mean: 336.47, median: 88.68, stdev: 352.50, best: 67.8976
#  gen: 124, stag: 21, mean: 364.05, median: 197.79, stdev: 323.78, best: 67.8976
#  gen: 125, stag: 22, mean: 234.89, median: 75.89, stdev: 311.29, best: 67.8976
#  gen: 126, stag: 23, mean: 312.10, median: 80.81, stdev: 373.67, best: 67.8976
#  gen: 127, stag: 24, mean: 263.68, median: 96.44, stdev: 314.28, best: 67.8976
#  gen: 128, stag: 25, mean: 225.37, median: 85.53, stdev: 280.45, best: 67.8976
#  gen: 129, stag: 0, mean: 327.27, median: 80.75, stdev: 485.00, best: 67.6514
#  gen: 130, stag: 1, mean: 818.34, median: 74.69, stdev: 5048.06, best: 67.6514
#  gen: 131, stag: 2, mean: 6472.14, median: 75.40, stdev: 21376.65, best: 67.6514
#  gen: 132, stag: 3, mean: 11550.47, median: 81.14, stdev: 26054.38, best: 67.6514
#  gen: 133, stag: 4, mean: 11305.03, median: 227.21, stdev: 20782.63, best: 67.6514
#  gen: 134, stag: 5, mean: 12118.56, median: 536.96, stdev: 22045.95, best: 67.6514
#  gen: 135, stag: 6, mean: 7922.98, median: 122.66, stdev: 18436.60, best: 67.6514
#  gen: 136, stag: 7, mean: 9588.29, median: 166.88, stdev: 25180.62, best: 67.6514
#  gen: 137, stag: 8, mean: 11454.55, median: 120.09, stdev: 25277.59, best: 67.6514
#  gen: 138, stag: 9, mean: 14888.82, median: 229.50, stdev: 27085.42, best: 67.6514
#  gen: 139, stag: 10, mean: 16393.56, median: 1062.93, stdev: 29276.11, best: 67.6514
#  gen: 140, stag: 11, mean: 13100.82, median: 123.58, stdev: 27102.33, best: 67.6514
#  gen: 141, stag: 12, mean: 20243.07, median: 669.23, stdev: 32320.52, best: 67.6514
#  gen: 142, stag: 13, mean: 7996.75, median: 83.45, stdev: 20879.16, best: 67.6514
#  gen: 143, stag: 14, mean: 12125.74, median: 490.43, stdev: 24471.21, best: 67.6514
#  gen: 144, stag: 15, mean: 13735.43, median: 561.00, stdev: 23469.83, best: 67.6514
#  gen: 145, stag: 16, mean: 14256.19, median: 467.27, stdev: 24716.74, best: 67.6514
#  gen: 146, stag: 0, mean: 17004.75, median: 99.35, stdev: 24554.40, best: 66.2028
#  gen: 147, stag: 1, mean: 13460.47, median: 74.31, stdev: 23194.42, best: 66.2028
#  gen: 148, stag: 2, mean: 9140.94, median: 101.44, stdev: 22064.98, best: 66.2028
#  gen: 149, stag: 3, mean: 4090.05, median: 88.53, stdev: 8133.50, best: 66.2028
#  gen: 150, stag: 4, mean: 5510.30, median: 111.24, stdev: 14407.32, best: 66.2028
#  gen: 151, stag: 5, mean: 3742.88, median: 212.41, stdev: 8697.18, best: 66.2028
#  gen: 152, stag: 6, mean: 3438.44, median: 75.19, stdev: 9262.83, best: 66.2028
#  gen: 153, stag: 0, mean: 2558.24, median: 77.15, stdev: 7898.38, best: 65.2258
#  gen: 154, stag: 1, mean: 4802.21, median: 85.52, stdev: 12887.04, best: 65.2258
#  gen: 155, stag: 2, mean: 468.45, median: 101.93, stdev: 1219.83, best: 65.2258
#  gen: 156, stag: 3, mean: 1352.80, median: 213.49, stdev: 2348.49, best: 65.2258
#  gen: 157, stag: 4, mean: 1705.53, median: 96.16, stdev: 3506.69, best: 65.2258
#  gen: 158, stag: 5, mean: 916.87, median: 262.04, stdev: 2422.00, best: 65.2258
#  gen: 159, stag: 6, mean: 1436.59, median: 74.00, stdev: 6441.78, best: 65.2258
#  gen: 160, stag: 7, mean: 6877.49, median: 110.81, stdev: 18072.18, best: 65.2258
#  gen: 161, stag: 8, mean: 5437.29, median: 99.30, stdev: 15739.32, best: 65.2258
#  gen: 162, stag: 0, mean: 5072.43, median: 70.65, stdev: 15755.25, best: 64.9014
#  gen: 163, stag: 1, mean: 1667.96, median: 72.36, stdev: 6469.74, best: 64.9014
#  gen: 164, stag: 2, mean: 1316.35, median: 84.95, stdev: 2896.14, best: 64.9014
#  gen: 165, stag: 3, mean: 2780.66, median: 88.25, stdev: 9031.60, best: 64.9014
#  gen: 166, stag: 4, mean: 7699.11, median: 109.84, stdev: 15993.70, best: 64.9014
#  gen: 167, stag: 5, mean: 8116.66, median: 111.99, stdev: 18504.70, best: 64.9014
#  gen: 168, stag: 6, mean: 4129.89, median: 98.25, stdev: 13061.76, best: 64.9014
#  gen: 169, stag: 7, mean: 1392.90, median: 98.91, stdev: 4279.92, best: 64.9014
#  gen: 170, stag: 0, mean: 1163.39, median: 74.74, stdev: 5277.67, best: 64.5578
#  gen: 171, stag: 1, mean: 2522.11, median: 85.71, stdev: 8993.75, best: 64.5578
#  gen: 172, stag: 2, mean: 3373.67, median: 300.41, stdev: 9402.66, best: 64.5578
#  gen: 173, stag: 3, mean: 1036.91, median: 251.87, stdev: 2375.88, best: 64.5578
#  gen: 174, stag: 4, mean: 838.42, median: 149.46, stdev: 1649.46, best: 64.5578
#  gen: 175, stag: 5, mean: 1542.48, median: 189.67, stdev: 2811.29, best: 64.5578
#  gen: 176, stag: 6, mean: 2285.48, median: 1177.28, stdev: 2995.04, best: 64.5578
#  gen: 177, stag: 7, mean: 2286.33, median: 288.51, stdev: 3055.78, best: 64.5578
#  gen: 178, stag: 8, mean: 1793.10, median: 70.92, stdev: 2959.78, best: 64.5578
#  gen: 179, stag: 9, mean: 2156.71, median: 69.63, stdev: 6117.58, best: 64.5578
#  gen: 180, stag: 10, mean: 2887.83, median: 84.76, stdev: 7655.86, best: 64.5578
#  gen: 181, stag: 11, mean: 1439.76, median: 77.05, stdev: 3841.03, best: 64.5578
#  gen: 182, stag: 12, mean: 680.03, median: 71.92, stdev: 1632.74, best: 64.5578
#  gen: 183, stag: 13, mean: 586.05, median: 95.50, stdev: 1019.83, best: 64.5578
#  gen: 184, stag: 14, mean: 443.46, median: 79.06, stdev: 728.56, best: 64.5578
#  gen: 185, stag: 15, mean: 486.25, median: 75.23, stdev: 731.71, best: 64.5578
#  gen: 186, stag: 16, mean: 822.50, median: 91.77, stdev: 1659.67, best: 64.5578
#  gen: 187, stag: 17, mean: 1555.15, median: 293.62, stdev: 5293.59, best: 64.5578
#  gen: 188, stag: 18, mean: 1483.26, median: 115.20, stdev: 2800.29, best: 64.5578
#  gen: 189, stag: 19, mean: 1238.05, median: 263.23, stdev: 2259.51, best: 64.5578
#  gen: 190, stag: 20, mean: 1251.46, median: 419.70, stdev: 2268.88, best: 64.5578
#  gen: 191, stag: 21, mean: 1612.15, median: 391.92, stdev: 2104.26, best: 64.5578
#  gen: 192, stag: 22, mean: 1139.13, median: 81.63, stdev: 2215.58, best: 64.5578
#  gen: 193, stag: 23, mean: 1275.98, median: 100.76, stdev: 1893.64, best: 64.5578
#  gen: 194, stag: 24, mean: 1536.91, median: 318.22, stdev: 2214.43, best: 64.5578
#  gen: 195, stag: 25, mean: 1683.28, median: 287.61, stdev: 2766.27, best: 64.5578
#  gen: 196, stag: 26, mean: 1709.63, median: 535.62, stdev: 2123.38, best: 64.5578
#  gen: 197, stag: 27, mean: 1929.87, median: 405.99, stdev: 2259.55, best: 64.5578
#  gen: 198, stag: 28, mean: 1927.55, median: 80.60, stdev: 2998.20, best: 64.5578
#  gen: 199, stag: 29, mean: 1819.09, median: 116.86, stdev: 3382.18, best: 64.5578
#  gen: 200, stag: 30, mean: 1941.46, median: 167.26, stdev: 2941.29, best: 64.5578
#  gen: 201, stag: 31, mean: 960.78, median: 104.26, stdev: 1674.16, best: 64.5578
#  gen: 202, stag: 32, mean: 1487.57, median: 151.29, stdev: 2803.97, best: 64.5578
#  gen: 203, stag: 33, mean: 1020.75, median: 77.03, stdev: 1691.41, best: 64.5578
#  gen: 204, stag: 0, mean: 625.34, median: 66.89, stdev: 1934.59, best: 64.4574
#  gen: 205, stag: 1, mean: 1026.91, median: 99.94, stdev: 4338.47, best: 64.4574
#  gen: 206, stag: 0, mean: 368.66, median: 66.20, stdev: 555.92, best: 64.3068
#  gen: 207, stag: 1, mean: 392.10, median: 88.42, stdev: 478.67, best: 64.3068
#  gen: 208, stag: 2, mean: 503.07, median: 158.73, stdev: 540.94, best: 64.3068
#  gen: 209, stag: 3, mean: 614.07, median: 105.01, stdev: 2625.40, best: 64.3068
#  gen: 210, stag: 4, mean: 756.86, median: 577.43, stdev: 2565.65, best: 64.3068
#  gen: 211, stag: 0, mean: 422.24, median: 74.36, stdev: 594.58, best: 64.3018
#  gen: 212, stag: 1, mean: 285.29, median: 66.18, stdev: 545.43, best: 64.3018
#  gen: 213, stag: 2, mean: 306.02, median: 67.72, stdev: 545.34, best: 64.3018
#  gen: 214, stag: 0, mean: 708.11, median: 65.66, stdev: 3828.43, best: 64.2874
#  gen: 215, stag: 0, mean: 377.52, median: 65.46, stdev: 970.02, best: 63.8782
#  gen: 216, stag: 1, mean: 514.97, median: 65.64, stdev: 1563.24, best: 63.8782
#  gen: 217, stag: 2, mean: 961.86, median: 67.39, stdev: 3645.49, best: 63.8782
#  gen: 218, stag: 0, mean: 2275.30, median: 65.33, stdev: 6679.09, best: 63.2161
#  gen: 219, stag: 1, mean: 2877.27, median: 75.35, stdev: 8856.96, best: 63.2161
#  gen: 220, stag: 2, mean: 274.31, median: 73.82, stdev: 531.19, best: 63.2161
#  gen: 221, stag: 3, mean: 187.85, median: 71.31, stdev: 200.29, best: 63.2161
#  gen: 222, stag: 4, mean: 202.94, median: 70.33, stdev: 392.13, best: 63.2161
#  gen: 223, stag: 5, mean: 127.00, median: 70.77, stdev: 106.08, best: 63.2161
#  gen: 224, stag: 6, mean: 117.77, median: 66.81, stdev: 97.69, best: 63.2161
#  gen: 225, stag: 7, mean: 85.41, median: 73.44, stdev: 37.75, best: 63.2161
#  gen: 226, stag: 8, mean: 395.46, median: 72.18, stdev: 3120.60, best: 63.2161
#  gen: 227, stag: 9, mean: 83.56, median: 64.90, stdev: 56.51, best: 63.2161
#  gen: 228, stag: 10, mean: 82.83, median: 72.88, stdev: 29.24, best: 63.2161
#  gen: 229, stag: 11, mean: 89.00, median: 68.66, stdev: 57.76, best: 63.2161
#  gen: 230, stag: 12, mean: 78.27, median: 65.25, stdev: 30.87, best: 63.2161
#  gen: 231, stag: 13, mean: 71.38, median: 64.67, stdev: 11.42, best: 63.2161
#  gen: 232, stag: 14, mean: 71.16, median: 65.18, stdev: 11.52, best: 63.2161
#  gen: 233, stag: 15, mean: 996.23, median: 64.94, stdev: 9207.69, best: 63.2161
#  gen: 234, stag: 16, mean: 65.82, median: 64.47, stdev: 2.51, best: 63.2161
#  gen: 235, stag: 17, mean: 81.03, median: 64.74, stdev: 139.10, best: 63.2161
#  gen: 236, stag: 18, mean: 67.92, median: 65.83, stdev: 6.91, best: 63.2161
#  gen: 237, stag: 19, mean: 72.04, median: 66.87, stdev: 14.48, best: 63.2161
#  gen: 238, stag: 20, mean: 94.59, median: 65.59, stdev: 274.64, best: 63.2161
#  gen: 239, stag: 21, mean: 75.68, median: 65.00, stdev: 85.33, best: 63.2161
#  gen: 240, stag: 22, mean: 65.84, median: 64.30, stdev: 2.66, best: 63.2161
#  gen: 241, stag: 23, mean: 68.36, median: 65.73, stdev: 13.75, best: 63.2161
#  gen: 242, stag: 24, mean: 127.83, median: 64.17, stdev: 490.61, best: 63.2161
#  gen: 243, stag: 25, mean: 284.88, median: 65.16, stdev: 1265.10, best: 63.2161
#  gen: 244, stag: 26, mean: 369.13, median: 66.15, stdev: 2124.19, best: 63.2161
#  gen: 245, stag: 27, mean: 538.52, median: 65.76, stdev: 2021.42, best: 63.2161
#  gen: 246, stag: 28, mean: 419.83, median: 65.87, stdev: 1196.12, best: 63.2161
#  gen: 247, stag: 29, mean: 279.51, median: 64.56, stdev: 711.62, best: 63.2161
#  gen: 248, stag: 30, mean: 304.90, median: 64.65, stdev: 644.36, best: 63.2161
#  gen: 249, stag: 31, mean: 1125.01, median: 65.78, stdev: 8183.86, best: 63.2161
#  gen: 250, stag: 32, mean: 324.06, median: 64.66, stdev: 522.06, best: 63.2161
#  gen: 251, stag: 33, mean: 301.14, median: 78.49, stdev: 561.84, best: 63.2161
#  gen: 252, stag: 34, mean: 178.75, median: 85.54, stdev: 196.84, best: 63.2161
#  gen: 253, stag: 35, mean: 525.91, median: 116.56, stdev: 3019.86, best: 63.2161
#  gen: 254, stag: 36, mean: 308.76, median: 129.68, stdev: 292.98, best: 63.2161
#  gen: 255, stag: 37, mean: 282.41, median: 96.12, stdev: 298.70, best: 63.2161
#  gen: 256, stag: 38, mean: 199.19, median: 76.25, stdev: 217.66, best: 63.2161
#  gen: 257, stag: 39, mean: 120.47, median: 64.43, stdev: 137.12, best: 63.2161
#  gen: 258, stag: 40, mean: 126.68, median: 63.88, stdev: 167.32, best: 63.2161
#  gen: 259, stag: 0, mean: 159.22, median: 63.75, stdev: 766.57, best: 62.8610
#  gen: 260, stag: 1, mean: 118.06, median: 63.72, stdev: 403.08, best: 62.8610
#  gen: 261, stag: 2, mean: 985.57, median: 64.73, stdev: 2811.09, best: 62.8610
#  gen: 262, stag: 3, mean: 679.21, median: 63.62, stdev: 2459.54, best: 62.8610
#  gen: 263, stag: 4, mean: 1878.19, median: 79.76, stdev: 4439.76, best: 62.8610
#  gen: 264, stag: 5, mean: 2445.84, median: 197.64, stdev: 8166.99, best: 62.8610
#  gen: 265, stag: 6, mean: 575.65, median: 74.29, stdev: 1796.43, best: 62.8610
#  gen: 266, stag: 7, mean: 1212.57, median: 123.50, stdev: 3725.57, best: 62.8610
#  gen: 267, stag: 8, mean: 1222.83, median: 77.79, stdev: 3338.12, best: 62.8610
#  gen: 268, stag: 9, mean: 251.51, median: 69.35, stdev: 517.23, best: 62.8610
#  gen: 269, stag: 10, mean: 149.24, median: 73.85, stdev: 212.80, best: 62.8610
#  gen: 270, stag: 11, mean: 238.01, median: 80.04, stdev: 353.23, best: 62.8610
#  gen: 271, stag: 12, mean: 284.08, median: 88.18, stdev: 328.19, best: 62.8610
#  gen: 272, stag: 13, mean: 1122.52, median: 65.92, stdev: 8607.50, best: 62.8610
#  gen: 273, stag: 14, mean: 995.25, median: 68.09, stdev: 7759.24, best: 62.8610
#  gen: 274, stag: 15, mean: 3837.97, median: 68.38, stdev: 15954.06, best: 62.8610
#  gen: 275, stag: 0, mean: 4022.98, median: 80.34, stdev: 14994.79, best: 57.7841
#  gen: 276, stag: 0, mean: 8056.89, median: 81.33, stdev: 20434.75, best: 53.2772
#  gen: 277, stag: 0, mean: 2669.70, median: 64.61, stdev: 6258.52, best: 52.1531
#  gen: 278, stag: 0, mean: 2260.31, median: 301.34, stdev: 3693.21, best: 46.9293
#  gen: 279, stag: 1, mean: 3146.95, median: 341.61, stdev: 4473.39, best: 46.9293
#  gen: 280, stag: 2, mean: 1164.98, median: 117.55, stdev: 2071.37, best: 46.9293
#  gen: 281, stag: 0, mean: 805.60, median: 63.35, stdev: 1420.30, best: 44.6290
#  gen: 282, stag: 1, mean: 713.73, median: 188.32, stdev: 930.50, best: 44.6290
#  gen: 283, stag: 2, mean: 492.83, median: 392.85, stdev: 591.53, best: 44.6290
#  gen: 284, stag: 3, mean: 349.73, median: 246.62, stdev: 359.05, best: 44.6290
#  gen: 285, stag: 4, mean: 290.45, median: 66.75, stdev: 358.30, best: 44.6290
#  gen: 286, stag: 5, mean: 343.74, median: 139.27, stdev: 429.45, best: 44.6290
#  gen: 287, stag: 6, mean: 443.90, median: 144.20, stdev: 656.17, best: 44.6290
#  gen: 288, stag: 7, mean: 1166.25, median: 63.53, stdev: 7957.38, best: 44.6290
#  gen: 289, stag: 8, mean: 347.78, median: 126.30, stdev: 551.21, best: 44.6290
#  gen: 290, stag: 9, mean: 1479.55, median: 136.75, stdev: 6069.00, best: 44.6290
#  gen: 291, stag: 10, mean: 1436.05, median: 70.23, stdev: 7148.46, best: 44.6290
#  gen: 292, stag: 11, mean: 694.77, median: 72.29, stdev: 1601.18, best: 44.6290
#  gen: 293, stag: 12, mean: 187.86, median: 72.86, stdev: 207.25, best: 44.6290
#  gen: 294, stag: 13, mean: 129.69, median: 76.57, stdev: 102.22, best: 44.6290
#  gen: 295, stag: 14, mean: 343.46, median: 135.90, stdev: 1751.84, best: 44.6290
#  gen: 296, stag: 15, mean: 146.66, median: 78.54, stdev: 118.94, best: 44.6290
#  gen: 297, stag: 16, mean: 145.41, median: 75.13, stdev: 136.64, best: 44.6290
#  gen: 298, stag: 17, mean: 1199.77, median: 61.63, stdev: 10413.82, best: 44.6290
#  gen: 299, stag: 18, mean: 822.55, median: 228.62, stdev: 4414.83, best: 44.6290
#  gen: 300, stag: 19, mean: 1211.82, median: 56.98, stdev: 5717.30, best: 44.6290
#  gen: 301, stag: 20, mean: 660.58, median: 53.95, stdev: 3820.27, best: 44.6290

#  Total generations: 301
#  Total CPU time: 6.225989459 s
#  Best score: 44.62897592287368
#  Real chromosome of best solution: [0.61998509 0.47713848 0.68980926 0.83734662 0.95167454 1.05824302
#  1.16365852]
#  Binary chromosome of best solution: []
