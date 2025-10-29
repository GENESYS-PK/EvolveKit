from evolvekit import *
from evolvekit.operators.Ga.real.mutation.UniformMutation import UniformMutation
from evolvekit.operators.Ga.real.crossover.DiscreteCrossover import DiscreteCrossover


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
crossover_real = DiscreteCrossover()
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
#  >>> EvolveKit Genetic Algorithm Demo <<<

#  gen: 1, stag: 0, mean: 95094.78, median: 85764.14, stdev: 63690.07, best: 6656.6039
#  gen: 2, stag: 1, mean: 70240.76, median: 59535.27, stdev: 40527.99, best: 6656.6039
#  gen: 3, stag: 2, mean: 77205.83, median: 87568.53, stdev: 41409.97, best: 6656.6039
#  gen: 4, stag: 3, mean: 82035.03, median: 87568.53, stdev: 43514.85, best: 6656.6039
#  gen: 5, stag: 4, mean: 82229.05, median: 92737.41, stdev: 45305.81, best: 6656.6039
#  gen: 6, stag: 5, mean: 90183.80, median: 132165.09, stdev: 48548.95, best: 6656.6039
#  gen: 7, stag: 6, mean: 92239.32, median: 132165.09, stdev: 49547.38, best: 6656.6039
#  gen: 8, stag: 7, mean: 79253.09, median: 48964.71, stdev: 49645.04, best: 6656.6039
#  gen: 9, stag: 8, mean: 69163.70, median: 48103.31, stdev: 46137.79, best: 6656.6039
#  gen: 10, stag: 9, mean: 64509.77, median: 49886.38, stdev: 43523.20, best: 6656.6039
#  gen: 11, stag: 10, mean: 36794.14, median: 39479.17, stdev: 13077.01, best: 6656.6039
#  gen: 12, stag: 11, mean: 31299.65, median: 29312.72, stdev: 12722.17, best: 6656.6039
#  gen: 13, stag: 12, mean: 30146.99, median: 29312.72, stdev: 11966.87, best: 6656.6039
#  gen: 14, stag: 13, mean: 30033.40, median: 29312.72, stdev: 12592.25, best: 6656.6039
#  gen: 15, stag: 14, mean: 30063.29, median: 29312.72, stdev: 13188.58, best: 6656.6039
#  gen: 16, stag: 15, mean: 29072.28, median: 29312.72, stdev: 12964.82, best: 6656.6039
#  gen: 17, stag: 16, mean: 28324.08, median: 18697.12, stdev: 14771.42, best: 6656.6039
#  gen: 18, stag: 17, mean: 24129.35, median: 18697.12, stdev: 12621.70, best: 6656.6039
#  gen: 19, stag: 18, mean: 24178.75, median: 18697.12, stdev: 13256.70, best: 6656.6039
#  gen: 20, stag: 19, mean: 26827.19, median: 18697.12, stdev: 14880.80, best: 6656.6039
#  gen: 21, stag: 20, mean: 24851.92, median: 18697.12, stdev: 14054.96, best: 6656.6039
#  gen: 22, stag: 21, mean: 32607.41, median: 29312.72, stdev: 16516.21, best: 6656.6039
#  gen: 23, stag: 22, mean: 33829.53, median: 29312.72, stdev: 21213.57, best: 6656.6039
#  gen: 24, stag: 23, mean: 33074.02, median: 49886.38, stdev: 17551.47, best: 6656.6039
#  gen: 25, stag: 24, mean: 30314.61, median: 15393.76, stdev: 17402.15, best: 6656.6039
#  gen: 26, stag: 25, mean: 25140.72, median: 15393.76, stdev: 15874.58, best: 6656.6039
#  gen: 27, stag: 26, mean: 21763.21, median: 15393.76, stdev: 14185.56, best: 6656.6039
#  gen: 28, stag: 27, mean: 26591.76, median: 15393.76, stdev: 16821.66, best: 6656.6039
#  gen: 29, stag: 28, mean: 21915.92, median: 15393.76, stdev: 14198.80, best: 6656.6039
#  gen: 30, stag: 29, mean: 25130.82, median: 15393.76, stdev: 16718.20, best: 6656.6039
#  gen: 31, stag: 30, mean: 23485.07, median: 15393.76, stdev: 15356.37, best: 6656.6039
#  gen: 32, stag: 31, mean: 15896.69, median: 15393.76, stdev: 6268.30, best: 6656.6039
#  gen: 33, stag: 32, mean: 19345.96, median: 15393.76, stdev: 11954.03, best: 6656.6039
#  gen: 34, stag: 33, mean: 19345.96, median: 15393.76, stdev: 11954.03, best: 6656.6039
#  gen: 35, stag: 34, mean: 15052.74, median: 15393.76, stdev: 2716.11, best: 6656.6039
#  gen: 36, stag: 35, mean: 15036.75, median: 15393.76, stdev: 2604.10, best: 6656.6039
#  gen: 37, stag: 36, mean: 15142.42, median: 15393.76, stdev: 3414.32, best: 6656.6039
#  gen: 38, stag: 37, mean: 14821.62, median: 15393.76, stdev: 1920.23, best: 6656.6039
#  gen: 39, stag: 38, mean: 14821.62, median: 15393.76, stdev: 1920.23, best: 6656.6039
#  gen: 40, stag: 39, mean: 14823.99, median: 15393.76, stdev: 1921.08, best: 6656.6039
#  gen: 41, stag: 40, mean: 14821.62, median: 15393.76, stdev: 1920.23, best: 6656.6039
#  gen: 42, stag: 41, mean: 14821.62, median: 15393.76, stdev: 1920.23, best: 6656.6039
#  gen: 43, stag: 42, mean: 14806.46, median: 15393.76, stdev: 2001.25, best: 6656.6039
#  gen: 44, stag: 43, mean: 14773.80, median: 15393.76, stdev: 1964.42, best: 6656.6039
#  gen: 45, stag: 44, mean: 14773.80, median: 15393.76, stdev: 1964.42, best: 6656.6039
#  gen: 46, stag: 45, mean: 14773.80, median: 15393.76, stdev: 1964.42, best: 6656.6039
#  gen: 47, stag: 46, mean: 14773.80, median: 15393.76, stdev: 1964.42, best: 6656.6039
#  gen: 48, stag: 47, mean: 14774.39, median: 15393.76, stdev: 1964.61, best: 6656.6039
#  gen: 49, stag: 48, mean: 14773.80, median: 15393.76, stdev: 1964.42, best: 6656.6039
#  gen: 50, stag: 49, mean: 14832.64, median: 15393.76, stdev: 2067.51, best: 6656.6039
#  gen: 51, stag: 50, mean: 14777.45, median: 15393.76, stdev: 1965.90, best: 6656.6039
#  gen: 52, stag: 51, mean: 14877.33, median: 15393.76, stdev: 2246.89, best: 6656.6039
#  gen: 53, stag: 52, mean: 14773.80, median: 15393.76, stdev: 1964.42, best: 6656.6039
#  gen: 54, stag: 53, mean: 14773.80, median: 15393.76, stdev: 1964.42, best: 6656.6039
#  gen: 55, stag: 54, mean: 14778.23, median: 15393.76, stdev: 1966.31, best: 6656.6039
#  gen: 56, stag: 55, mean: 14862.32, median: 15393.76, stdev: 2424.83, best: 6656.6039
#  gen: 57, stag: 56, mean: 14726.14, median: 15393.76, stdev: 1996.17, best: 6656.6039
#  gen: 58, stag: 57, mean: 14726.14, median: 15393.76, stdev: 1996.17, best: 6656.6039
#  gen: 59, stag: 58, mean: 14726.14, median: 15393.76, stdev: 1996.17, best: 6656.6039
#  gen: 60, stag: 59, mean: 14730.27, median: 15393.76, stdev: 1997.97, best: 6656.6039
#  gen: 61, stag: 60, mean: 14726.14, median: 15393.76, stdev: 1996.17, best: 6656.6039
#  gen: 62, stag: 61, mean: 14713.91, median: 15393.76, stdev: 1992.50, best: 6656.6039
#  gen: 63, stag: 62, mean: 14698.63, median: 15393.76, stdev: 1988.21, best: 6656.6039
#  gen: 64, stag: 63, mean: 14219.86, median: 15393.76, stdev: 2516.49, best: 6656.6039
#  gen: 65, stag: 64, mean: 14204.56, median: 15393.76, stdev: 2509.77, best: 6656.6039
#  gen: 66, stag: 65, mean: 14156.27, median: 15240.77, stdev: 2504.82, best: 6656.6039
#  gen: 67, stag: 66, mean: 14028.99, median: 15240.77, stdev: 3188.01, best: 6656.6039
#  gen: 68, stag: 67, mean: 13497.91, median: 15240.77, stdev: 2854.50, best: 6656.6039
#  gen: 69, stag: 68, mean: 14696.36, median: 15240.77, stdev: 8237.85, best: 6656.6039
#  gen: 70, stag: 69, mean: 13349.43, median: 15240.77, stdev: 4410.90, best: 6656.6039
#  gen: 71, stag: 70, mean: 12865.38, median: 15317.26, stdev: 3087.31, best: 6656.6039
#  gen: 72, stag: 71, mean: 12882.67, median: 15240.77, stdev: 3085.76, best: 6656.6039
#  gen: 73, stag: 72, mean: 12417.97, median: 11222.73, stdev: 3036.82, best: 6656.6039
#  gen: 74, stag: 73, mean: 12267.14, median: 10611.87, stdev: 4208.84, best: 6656.6039
#  gen: 75, stag: 74, mean: 11030.28, median: 10611.87, stdev: 2698.36, best: 6656.6039
#  gen: 76, stag: 75, mean: 10313.00, median: 10611.87, stdev: 1984.17, best: 6656.6039
#  gen: 77, stag: 76, mean: 10523.68, median: 10611.87, stdev: 2618.31, best: 6656.6039
#  gen: 78, stag: 0, mean: 10373.46, median: 8554.15, stdev: 2866.91, best: 5936.6004
#  gen: 79, stag: 1, mean: 10817.16, median: 8554.15, stdev: 3304.33, best: 5936.6004
#  gen: 80, stag: 2, mean: 9865.48, median: 8554.15, stdev: 2788.26, best: 5936.6004
#  gen: 81, stag: 3, mean: 10834.75, median: 8554.15, stdev: 3293.77, best: 5936.6004
#  gen: 82, stag: 4, mean: 10556.06, median: 8554.15, stdev: 3541.62, best: 5936.6004
#  gen: 83, stag: 5, mean: 11364.32, median: 8554.15, stdev: 3447.48, best: 5936.6004
#  gen: 84, stag: 6, mean: 11465.66, median: 8554.15, stdev: 3590.05, best: 5936.6004
#  gen: 85, stag: 7, mean: 12575.70, median: 8554.15, stdev: 12011.86, best: 5936.6004
#  gen: 86, stag: 8, mean: 12007.53, median: 15238.79, stdev: 3613.76, best: 5936.6004
#  gen: 87, stag: 9, mean: 11460.07, median: 8554.15, stdev: 3560.51, best: 5936.6004
#  gen: 88, stag: 10, mean: 11656.34, median: 8656.32, stdev: 3591.24, best: 5936.6004
#  gen: 89, stag: 11, mean: 11397.26, median: 8554.15, stdev: 3579.70, best: 5936.6004
#  gen: 90, stag: 12, mean: 11671.44, median: 8554.15, stdev: 3613.78, best: 5936.6004
#  gen: 91, stag: 13, mean: 10710.25, median: 8554.15, stdev: 3486.96, best: 5936.6004
#  gen: 92, stag: 14, mean: 11531.01, median: 8554.15, stdev: 3682.34, best: 5936.6004
#  gen: 93, stag: 15, mean: 11462.61, median: 8554.15, stdev: 3673.46, best: 5936.6004
#  gen: 94, stag: 16, mean: 10923.37, median: 8554.15, stdev: 3550.23, best: 5936.6004
#  gen: 95, stag: 17, mean: 11288.73, median: 8554.15, stdev: 10298.84, best: 5936.6004
#  gen: 96, stag: 18, mean: 10496.21, median: 8554.15, stdev: 3410.95, best: 5936.6004
#  gen: 97, stag: 19, mean: 10652.68, median: 8554.15, stdev: 9122.34, best: 5936.6004
#  gen: 98, stag: 20, mean: 10176.89, median: 8554.15, stdev: 8153.33, best: 5936.6004
#  gen: 99, stag: 21, mean: 10490.19, median: 8554.15, stdev: 3432.71, best: 5936.6004
#  gen: 100, stag: 22, mean: 10247.30, median: 8554.15, stdev: 3216.55, best: 5936.6004
#  gen: 101, stag: 23, mean: 10175.54, median: 8554.15, stdev: 3102.84, best: 5936.6004
#  gen: 102, stag: 24, mean: 11272.56, median: 8332.89, stdev: 7103.19, best: 5936.6004
#  gen: 103, stag: 25, mean: 9407.77, median: 8332.89, stdev: 3249.51, best: 5936.6004
#  gen: 104, stag: 26, mean: 8969.49, median: 6823.27, stdev: 3276.64, best: 5936.6004
#  gen: 105, stag: 27, mean: 10134.78, median: 6823.27, stdev: 4055.57, best: 5936.6004
#  gen: 106, stag: 28, mean: 11612.21, median: 15393.76, stdev: 4267.08, best: 5936.6004
#  gen: 107, stag: 29, mean: 10293.54, median: 6823.27, stdev: 4304.09, best: 5936.6004
#  gen: 108, stag: 30, mean: 10069.52, median: 6823.27, stdev: 4169.20, best: 5936.6004
#  gen: 109, stag: 31, mean: 9212.48, median: 6823.27, stdev: 3855.74, best: 5936.6004
#  gen: 110, stag: 32, mean: 9346.99, median: 6823.27, stdev: 3895.98, best: 5936.6004
#  gen: 111, stag: 0, mean: 9495.45, median: 6823.27, stdev: 4095.55, best: 693.3832
#  gen: 112, stag: 1, mean: 9322.59, median: 6823.27, stdev: 4021.60, best: 693.3832
#  gen: 113, stag: 2, mean: 8334.52, median: 6823.27, stdev: 3803.05, best: 693.3832
#  gen: 114, stag: 3, mean: 6751.44, median: 6823.27, stdev: 615.41, best: 693.3832
#  gen: 115, stag: 4, mean: 6715.55, median: 6823.27, stdev: 707.90, best: 693.3832
#  gen: 116, stag: 5, mean: 6715.55, median: 6823.27, stdev: 707.90, best: 693.3832
#  gen: 117, stag: 6, mean: 6715.55, median: 6823.27, stdev: 707.90, best: 693.3832
#  gen: 118, stag: 7, mean: 6715.55, median: 6823.27, stdev: 707.90, best: 693.3832
#  gen: 119, stag: 8, mean: 6716.28, median: 6823.27, stdev: 708.05, best: 693.3832
#  gen: 120, stag: 9, mean: 6715.55, median: 6823.27, stdev: 707.90, best: 693.3832
#  gen: 121, stag: 10, mean: 6715.02, median: 6823.27, stdev: 707.84, best: 693.3832
#  gen: 122, stag: 11, mean: 6715.02, median: 6823.27, stdev: 707.84, best: 693.3832
#  gen: 123, stag: 12, mean: 6693.23, median: 6823.27, stdev: 737.12, best: 693.3832
#  gen: 124, stag: 13, mean: 6693.23, median: 6823.27, stdev: 737.12, best: 693.3832
#  gen: 125, stag: 14, mean: 6702.71, median: 6823.27, stdev: 744.79, best: 693.3832
#  gen: 126, stag: 15, mean: 6693.23, median: 6823.27, stdev: 737.12, best: 693.3832
#  gen: 127, stag: 16, mean: 6575.34, median: 6823.27, stdev: 1092.93, best: 693.3832
#  gen: 128, stag: 17, mean: 6359.98, median: 6823.27, stdev: 1346.96, best: 693.3832
#  gen: 129, stag: 18, mean: 6356.35, median: 6823.27, stdev: 1346.20, best: 693.3832
#  gen: 130, stag: 19, mean: 6481.47, median: 6823.27, stdev: 2486.72, best: 693.3832
#  gen: 131, stag: 20, mean: 6275.93, median: 6823.27, stdev: 1342.20, best: 693.3832
#  gen: 132, stag: 21, mean: 6187.27, median: 6823.27, stdev: 1332.36, best: 693.3832
#  gen: 133, stag: 22, mean: 6203.69, median: 6823.27, stdev: 1912.57, best: 693.3832
#  gen: 134, stag: 23, mean: 6777.20, median: 6823.27, stdev: 3323.51, best: 693.3832
#  gen: 135, stag: 24, mean: 6016.37, median: 5936.60, stdev: 1297.76, best: 693.3832
#  gen: 136, stag: 25, mean: 5998.64, median: 5936.60, stdev: 1292.66, best: 693.3832
#  gen: 137, stag: 26, mean: 5909.97, median: 5936.60, stdev: 1263.12, best: 693.3832
#  gen: 138, stag: 27, mean: 6025.24, median: 6379.94, stdev: 1300.21, best: 693.3832
#  gen: 139, stag: 28, mean: 6051.84, median: 6823.27, stdev: 1307.17, best: 693.3832
#  gen: 140, stag: 29, mean: 6073.96, median: 6823.27, stdev: 1314.74, best: 693.3832
#  gen: 141, stag: 30, mean: 5998.64, median: 5936.60, stdev: 1292.66, best: 693.3832
#  gen: 142, stag: 31, mean: 6016.37, median: 5936.60, stdev: 1297.76, best: 693.3832
#  gen: 143, stag: 32, mean: 6113.91, median: 6823.27, stdev: 1321.20, best: 693.3832
#  gen: 144, stag: 33, mean: 6017.92, median: 5936.60, stdev: 5640.50, best: 693.3832
#  gen: 145, stag: 34, mean: 5269.72, median: 5936.60, stdev: 1734.62, best: 693.3832
#  gen: 146, stag: 35, mean: 4937.53, median: 3938.88, stdev: 2053.28, best: 693.3832
#  gen: 147, stag: 36, mean: 6374.84, median: 6379.94, stdev: 8932.15, best: 693.3832
#  gen: 148, stag: 37, mean: 8282.78, median: 6823.27, stdev: 10664.29, best: 693.3832
#  gen: 149, stag: 38, mean: 7581.08, median: 6432.23, stdev: 10133.91, best: 693.3832
#  gen: 150, stag: 39, mean: 8103.31, median: 6823.27, stdev: 10717.41, best: 693.3832
#  gen: 151, stag: 40, mean: 11375.55, median: 3233.95, stdev: 16659.37, best: 693.3832
#  gen: 152, stag: 41, mean: 10013.96, median: 3233.95, stdev: 14011.60, best: 693.3832
#  gen: 153, stag: 42, mean: 7359.68, median: 3233.95, stdev: 10546.13, best: 693.3832
#  gen: 154, stag: 43, mean: 9977.62, median: 3233.95, stdev: 14323.68, best: 693.3832
#  gen: 155, stag: 44, mean: 14312.00, median: 3233.95, stdev: 17918.13, best: 693.3832
#  gen: 156, stag: 45, mean: 11791.17, median: 3233.95, stdev: 16267.96, best: 693.3832
#  gen: 157, stag: 46, mean: 10878.58, median: 3233.95, stdev: 15616.25, best: 693.3832
#  gen: 158, stag: 47, mean: 13461.14, median: 3233.95, stdev: 17888.25, best: 693.3832
#  gen: 159, stag: 48, mean: 14271.31, median: 3233.95, stdev: 17939.69, best: 693.3832
#  gen: 160, stag: 49, mean: 23094.15, median: 3233.95, stdev: 21426.76, best: 693.3832
#  gen: 161, stag: 50, mean: 16167.61, median: 3233.95, stdev: 19465.64, best: 693.3832
#  gen: 162, stag: 51, mean: 15342.19, median: 3233.95, stdev: 19118.20, best: 693.3832
#  gen: 163, stag: 52, mean: 19125.98, median: 3233.95, stdev: 20460.62, best: 693.3832
#  gen: 164, stag: 53, mean: 20387.16, median: 3233.95, stdev: 20735.96, best: 693.3832
#  gen: 165, stag: 54, mean: 19546.88, median: 3233.95, stdev: 20562.04, best: 693.3832
#  gen: 166, stag: 55, mean: 8200.85, median: 3233.95, stdev: 13691.31, best: 693.3832
#  gen: 167, stag: 56, mean: 13234.33, median: 3233.95, stdev: 18002.00, best: 693.3832
#  gen: 168, stag: 57, mean: 12018.49, median: 3233.95, stdev: 17259.94, best: 693.3832
#  gen: 169, stag: 58, mean: 3147.95, median: 3233.95, stdev: 419.50, best: 693.3832
#  gen: 170, stag: 59, mean: 3178.51, median: 3233.95, stdev: 476.70, best: 693.3832
#  gen: 171, stag: 60, mean: 3147.95, median: 3233.95, stdev: 419.50, best: 693.3832
#  gen: 172, stag: 61, mean: 3445.71, median: 3233.95, stdev: 3000.76, best: 693.3832
#  gen: 173, stag: 62, mean: 3147.95, median: 3233.95, stdev: 419.50, best: 693.3832
#  gen: 174, stag: 63, mean: 3328.78, median: 3233.95, stdev: 1461.34, best: 693.3832
#  gen: 175, stag: 64, mean: 3147.95, median: 3233.95, stdev: 419.50, best: 693.3832
#  gen: 176, stag: 65, mean: 3823.58, median: 3233.95, stdev: 6744.15, best: 693.3832
#  gen: 177, stag: 66, mean: 3147.95, median: 3233.95, stdev: 419.50, best: 693.3832
#  gen: 178, stag: 67, mean: 3144.33, median: 3233.95, stdev: 420.30, best: 693.3832
#  gen: 179, stag: 68, mean: 3200.04, median: 3233.95, stdev: 804.48, best: 693.3832
#  gen: 180, stag: 69, mean: 3132.91, median: 3233.95, stdev: 433.03, best: 693.3832
#  gen: 181, stag: 70, mean: 3132.91, median: 3233.95, stdev: 433.03, best: 693.3832
#  gen: 182, stag: 71, mean: 3326.09, median: 3233.95, stdev: 1815.35, best: 693.3832
#  gen: 183, stag: 72, mean: 3218.99, median: 3233.95, stdev: 953.91, best: 693.3832
#  gen: 184, stag: 73, mean: 3565.89, median: 3233.95, stdev: 1913.25, best: 693.3832
#  gen: 185, stag: 74, mean: 3132.91, median: 3233.95, stdev: 433.03, best: 693.3832
#  gen: 186, stag: 75, mean: 3132.91, median: 3233.95, stdev: 433.03, best: 693.3832
#  gen: 187, stag: 76, mean: 3130.21, median: 3233.95, stdev: 433.24, best: 693.3832
#  gen: 188, stag: 77, mean: 3329.74, median: 3233.95, stdev: 2042.21, best: 693.3832
#  gen: 189, stag: 78, mean: 3071.97, median: 3233.95, stdev: 478.27, best: 693.3832
#  gen: 190, stag: 79, mean: 3083.22, median: 3233.95, stdev: 494.89, best: 693.3832
#  gen: 191, stag: 80, mean: 3071.97, median: 3233.95, stdev: 478.27, best: 693.3832
#  gen: 192, stag: 81, mean: 3049.00, median: 3233.95, stdev: 523.02, best: 693.3832
#  gen: 193, stag: 82, mean: 3266.92, median: 3233.95, stdev: 2248.48, best: 693.3832
#  gen: 194, stag: 83, mean: 3049.00, median: 3233.95, stdev: 523.02, best: 693.3832
#  gen: 195, stag: 84, mean: 3049.00, median: 3233.95, stdev: 523.02, best: 693.3832
#  gen: 196, stag: 85, mean: 3049.00, median: 3233.95, stdev: 523.02, best: 693.3832
#  gen: 197, stag: 86, mean: 3224.69, median: 3233.95, stdev: 1842.39, best: 693.3832
#  gen: 198, stag: 87, mean: 3049.00, median: 3233.95, stdev: 523.02, best: 693.3832
#  gen: 199, stag: 88, mean: 3049.00, median: 3233.95, stdev: 523.02, best: 693.3832
#  gen: 200, stag: 89, mean: 3197.72, median: 3233.95, stdev: 1538.77, best: 693.3832
#  gen: 201, stag: 90, mean: 3049.00, median: 3233.95, stdev: 523.02, best: 693.3832
#  gen: 202, stag: 91, mean: 3049.00, median: 3233.95, stdev: 523.02, best: 693.3832
#  gen: 203, stag: 92, mean: 3047.25, median: 3233.95, stdev: 522.69, best: 693.3832
#  gen: 204, stag: 93, mean: 3047.25, median: 3233.95, stdev: 522.69, best: 693.3832
#  gen: 205, stag: 94, mean: 3047.25, median: 3233.95, stdev: 522.69, best: 693.3832
#  gen: 206, stag: 95, mean: 3047.25, median: 3233.95, stdev: 522.69, best: 693.3832
#  gen: 207, stag: 96, mean: 3047.25, median: 3233.95, stdev: 522.69, best: 693.3832
#  gen: 208, stag: 97, mean: 3099.58, median: 3233.95, stdev: 750.88, best: 693.3832
#  gen: 209, stag: 98, mean: 3151.91, median: 3233.95, stdev: 921.38, best: 693.3832
#  gen: 210, stag: 99, mean: 3047.25, median: 3233.95, stdev: 522.69, best: 693.3832
#  gen: 211, stag: 100, mean: 2959.90, median: 3233.95, stdev: 563.69, best: 693.3832
#  gen: 212, stag: 101, mean: 2959.90, median: 3233.95, stdev: 563.69, best: 693.3832
#  gen: 213, stag: 102, mean: 2959.90, median: 3233.95, stdev: 563.69, best: 693.3832
#  gen: 214, stag: 103, mean: 3228.94, median: 3233.95, stdev: 3044.76, best: 693.3832
#  gen: 215, stag: 104, mean: 2823.27, median: 3059.76, stdev: 567.65, best: 693.3832
#  gen: 216, stag: 105, mean: 2707.39, median: 2872.10, stdev: 564.99, best: 693.3832
#  gen: 217, stag: 106, mean: 2754.03, median: 2872.10, stdev: 944.38, best: 693.3832
#  gen: 218, stag: 107, mean: 2530.48, median: 2263.35, stdev: 567.00, best: 693.3832
#  gen: 219, stag: 108, mean: 2437.80, median: 2263.35, stdev: 524.21, best: 693.3832
#  gen: 220, stag: 109, mean: 2749.59, median: 2263.35, stdev: 2465.91, best: 693.3832
#  gen: 221, stag: 110, mean: 2652.18, median: 2263.35, stdev: 1851.60, best: 693.3832
#  gen: 222, stag: 111, mean: 2544.57, median: 2263.35, stdev: 574.26, best: 693.3832
#  gen: 223, stag: 112, mean: 2742.54, median: 2263.35, stdev: 1932.04, best: 693.3832
#  gen: 224, stag: 113, mean: 2501.25, median: 2263.35, stdev: 1769.78, best: 693.3832
#  gen: 225, stag: 114, mean: 2357.87, median: 2263.35, stdev: 482.09, best: 693.3832
#  gen: 226, stag: 115, mean: 2339.09, median: 2263.35, stdev: 641.37, best: 693.3832
#  gen: 227, stag: 116, mean: 2469.69, median: 2263.35, stdev: 569.90, best: 693.3832
#  gen: 228, stag: 117, mean: 2337.25, median: 2263.35, stdev: 466.03, best: 693.3832
#  gen: 229, stag: 118, mean: 2540.34, median: 2263.35, stdev: 2630.33, best: 693.3832
#  gen: 230, stag: 119, mean: 2984.44, median: 2263.35, stdev: 3751.80, best: 693.3832
#  gen: 231, stag: 120, mean: 3295.32, median: 2263.35, stdev: 2884.77, best: 693.3832
#  gen: 232, stag: 121, mean: 3323.13, median: 2263.35, stdev: 2593.25, best: 693.3832
#  gen: 233, stag: 122, mean: 2891.86, median: 2263.35, stdev: 2004.63, best: 693.3832
#  gen: 234, stag: 123, mean: 2447.47, median: 2263.35, stdev: 588.17, best: 693.3832
#  gen: 235, stag: 124, mean: 2533.84, median: 2252.14, stdev: 786.37, best: 693.3832
#  gen: 236, stag: 125, mean: 2535.96, median: 2252.14, stdev: 881.89, best: 693.3832
#  gen: 237, stag: 126, mean: 2489.74, median: 2263.35, stdev: 623.89, best: 693.3832
#  gen: 238, stag: 127, mean: 2436.07, median: 2092.06, stdev: 603.68, best: 693.3832
#  gen: 239, stag: 128, mean: 2573.10, median: 2263.35, stdev: 638.63, best: 693.3832
#  gen: 240, stag: 129, mean: 2620.72, median: 2263.35, stdev: 699.24, best: 693.3832
#  gen: 241, stag: 130, mean: 2613.06, median: 2263.35, stdev: 655.79, best: 693.3832
#  gen: 242, stag: 131, mean: 2353.85, median: 2092.06, stdev: 577.63, best: 693.3832
#  gen: 243, stag: 132, mean: 2234.24, median: 2092.06, stdev: 480.10, best: 693.3832
#  gen: 244, stag: 133, mean: 2748.49, median: 2092.06, stdev: 3607.86, best: 693.3832
#  gen: 245, stag: 134, mean: 2447.35, median: 2263.35, stdev: 598.96, best: 693.3832
#  gen: 246, stag: 135, mean: 2568.13, median: 2092.06, stdev: 756.24, best: 693.3832
#  gen: 247, stag: 136, mean: 2658.51, median: 2092.06, stdev: 989.10, best: 693.3832
#  gen: 248, stag: 137, mean: 2776.92, median: 3233.95, stdev: 1072.07, best: 693.3832
#  gen: 249, stag: 138, mean: 2380.49, median: 2092.06, stdev: 693.75, best: 693.3832
#  gen: 250, stag: 139, mean: 2330.73, median: 2092.06, stdev: 593.10, best: 693.3832
#  gen: 251, stag: 140, mean: 2262.21, median: 2092.06, stdev: 549.12, best: 693.3832
#  gen: 252, stag: 141, mean: 2216.54, median: 2092.06, stdev: 512.68, best: 693.3832
#  gen: 253, stag: 142, mean: 2234.97, median: 2092.06, stdev: 540.26, best: 693.3832
#  gen: 254, stag: 143, mean: 2159.44, median: 2092.06, stdev: 456.72, best: 693.3832
#  gen: 255, stag: 144, mean: 2033.83, median: 2092.06, stdev: 257.52, best: 693.3832
#  gen: 256, stag: 145, mean: 2033.83, median: 2092.06, stdev: 257.52, best: 693.3832
#  gen: 257, stag: 146, mean: 2033.83, median: 2092.06, stdev: 257.52, best: 693.3832
#  gen: 258, stag: 147, mean: 2195.24, median: 2092.06, stdev: 1400.48, best: 693.3832
#  gen: 259, stag: 148, mean: 2711.85, median: 2092.06, stdev: 2979.88, best: 693.3832
#  gen: 260, stag: 149, mean: 3924.70, median: 2092.06, stdev: 4717.49, best: 693.3832
#  gen: 261, stag: 150, mean: 4393.19, median: 2092.06, stdev: 5272.04, best: 693.3832
#  gen: 262, stag: 151, mean: 4067.89, median: 2092.06, stdev: 4873.22, best: 693.3832
#  gen: 263, stag: 152, mean: 4769.74, median: 2092.06, stdev: 8351.20, best: 693.3832
#  gen: 264, stag: 153, mean: 10856.79, median: 2092.06, stdev: 21113.04, best: 693.3832
#  gen: 265, stag: 154, mean: 12562.55, median: 2092.06, stdev: 25086.74, best: 693.3832
#  gen: 266, stag: 155, mean: 11211.60, median: 2092.06, stdev: 23633.78, best: 693.3832
#  gen: 267, stag: 156, mean: 11101.45, median: 2092.06, stdev: 23650.35, best: 693.3832
#  gen: 268, stag: 157, mean: 27946.03, median: 2092.06, stdev: 33975.44, best: 693.3832
#  gen: 269, stag: 158, mean: 28647.89, median: 2092.06, stdev: 34158.55, best: 693.3832
#  gen: 270, stag: 159, mean: 23033.03, median: 2092.06, stdev: 32239.90, best: 693.3832
#  gen: 271, stag: 160, mean: 39175.75, median: 72277.82, stdev: 35152.87, best: 693.3832
#  gen: 272, stag: 161, mean: 30051.60, median: 2092.06, stdev: 34479.03, best: 693.3832
#  gen: 273, stag: 162, mean: 13302.90, median: 2092.06, stdev: 25759.00, best: 693.3832
#  gen: 274, stag: 163, mean: 12411.82, median: 2092.06, stdev: 24335.69, best: 693.3832
#  gen: 275, stag: 164, mean: 1892.49, median: 2092.06, stdev: 403.81, best: 693.3832
#  gen: 276, stag: 165, mean: 1936.87, median: 2092.06, stdev: 612.95, best: 693.3832
#  gen: 277, stag: 166, mean: 1892.49, median: 2092.06, stdev: 403.81, best: 693.3832
#  gen: 278, stag: 167, mean: 1798.26, median: 2092.06, stdev: 453.17, best: 693.3832
#  gen: 279, stag: 168, mean: 2073.52, median: 2092.06, stdev: 2806.46, best: 693.3832
#  gen: 280, stag: 169, mean: 3365.75, median: 2092.06, stdev: 6652.72, best: 693.3832
#  gen: 281, stag: 170, mean: 5289.80, median: 2092.06, stdev: 9420.79, best: 693.3832
#  gen: 282, stag: 171, mean: 5858.83, median: 2092.06, stdev: 12585.97, best: 693.3832
#  gen: 283, stag: 172, mean: 1521.63, median: 1149.77, stdev: 491.67, best: 693.3832
#  gen: 284, stag: 173, mean: 1446.25, median: 1149.77, stdev: 470.20, best: 693.3832
#  gen: 285, stag: 174, mean: 1314.33, median: 1149.77, stdev: 396.99, best: 693.3832
#  gen: 286, stag: 175, mean: 1568.75, median: 1149.77, stdev: 498.87, best: 693.3832
#  gen: 287, stag: 176, mean: 1653.55, median: 2092.06, stdev: 500.53, best: 693.3832
#  gen: 288, stag: 177, mean: 1606.44, median: 2092.06, stdev: 501.38, best: 693.3832
#  gen: 289, stag: 178, mean: 1539.84, median: 1149.77, stdev: 494.42, best: 693.3832
#  gen: 290, stag: 179, mean: 1407.83, median: 1149.77, stdev: 466.73, best: 693.3832
#  gen: 291, stag: 180, mean: 1379.56, median: 1149.77, stdev: 452.76, best: 693.3832
#  gen: 292, stag: 181, mean: 1532.49, median: 1149.77, stdev: 2207.32, best: 693.3832
#  gen: 293, stag: 182, mean: 1723.83, median: 1149.77, stdev: 3008.91, best: 693.3832
#  gen: 294, stag: 183, mean: 3112.84, median: 1149.77, stdev: 6223.16, best: 693.3832
#  gen: 295, stag: 184, mean: 3033.33, median: 1149.77, stdev: 5533.49, best: 693.3832
#  gen: 296, stag: 185, mean: 1941.18, median: 1149.77, stdev: 2998.92, best: 693.3832
#  gen: 297, stag: 186, mean: 1456.40, median: 1149.77, stdev: 527.30, best: 693.3832
#  gen: 298, stag: 187, mean: 1391.57, median: 1143.03, stdev: 542.85, best: 693.3832
#  gen: 299, stag: 188, mean: 1390.66, median: 936.64, stdev: 563.17, best: 693.3832
#  gen: 300, stag: 189, mean: 1336.46, median: 936.64, stdev: 555.38, best: 693.3832
#  gen: 301, stag: 190, mean: 1220.91, median: 936.64, stdev: 503.97, best: 693.3832

#  Total generations: 301
#  Total CPU time: 7.306286735 s
#  Best score: 693.383199349442
#  Real chromosome of best solution: [-0.51186496  1.65189366  1.02763376  0.94883183  0.23654799  0.38172787
#   1.37587211]
#  Binary chromosome of best solution: []
