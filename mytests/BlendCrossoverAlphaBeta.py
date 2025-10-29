from evolvekit import *
from evolvekit.operators.Ga.real.mutation.UniformMutation import UniformMutation
from evolvekit.operators.Ga.real.crossover.BlendCrossoverAlphaBeta import (
    BlendCrossoverAlphaBeta,
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
crossover_real = BlendCrossoverAlphaBeta()
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
#  gen: 2, stag: 1, mean: 63613.50, median: 55551.29, stdev: 38812.10, best: 6656.6039
#  gen: 3, stag: 0, mean: 58967.16, median: 43702.76, stdev: 43179.17, best: 6081.0466
#  gen: 4, stag: 0, mean: 54696.13, median: 32767.61, stdev: 48077.10, best: 2942.1932
#  gen: 5, stag: 1, mean: 45847.65, median: 24708.31, stdev: 43403.83, best: 2942.1932
#  gen: 6, stag: 2, mean: 52630.93, median: 23273.14, stdev: 53822.59, best: 2942.1932
#  gen: 7, stag: 3, mean: 56688.49, median: 31800.93, stdev: 54073.41, best: 2942.1932
#  gen: 8, stag: 4, mean: 55584.85, median: 43727.76, stdev: 50202.85, best: 2942.1932
#  gen: 9, stag: 5, mean: 52594.62, median: 48475.75, stdev: 41643.98, best: 2942.1932
#  gen: 10, stag: 6, mean: 58863.13, median: 44937.02, stdev: 51041.75, best: 2942.1932
#  gen: 11, stag: 7, mean: 61130.70, median: 46206.25, stdev: 53391.30, best: 2942.1932
#  gen: 12, stag: 8, mean: 59959.93, median: 23195.15, stdev: 58762.67, best: 2942.1932
#  gen: 13, stag: 9, mean: 68962.56, median: 66736.88, stdev: 64012.60, best: 2942.1932
#  gen: 14, stag: 10, mean: 48596.32, median: 14360.51, stdev: 61159.98, best: 2942.1932
#  gen: 15, stag: 11, mean: 74028.10, median: 33877.78, stdev: 78572.15, best: 2942.1932
#  gen: 16, stag: 12, mean: 66106.44, median: 46700.31, stdev: 69094.36, best: 2942.1932
#  gen: 17, stag: 13, mean: 54458.05, median: 37724.64, stdev: 54027.86, best: 2942.1932
#  gen: 18, stag: 14, mean: 66288.00, median: 55068.05, stdev: 61059.44, best: 2942.1932
#  gen: 19, stag: 15, mean: 66336.45, median: 42330.66, stdev: 64577.56, best: 2942.1932
#  gen: 20, stag: 0, mean: 59863.34, median: 19001.06, stdev: 66433.70, best: 1331.9891
#  gen: 21, stag: 1, mean: 43415.79, median: 12344.56, stdev: 52209.41, best: 1331.9891
#  gen: 22, stag: 2, mean: 36164.86, median: 8818.95, stdev: 44146.22, best: 1331.9891
#  gen: 23, stag: 3, mean: 42402.59, median: 12035.72, stdev: 51545.80, best: 1331.9891
#  gen: 24, stag: 4, mean: 54461.14, median: 20305.67, stdev: 59167.10, best: 1331.9891
#  gen: 25, stag: 5, mean: 66000.53, median: 32073.38, stdev: 72044.78, best: 1331.9891
#  gen: 26, stag: 0, mean: 49082.47, median: 7267.79, stdev: 60231.89, best: 1033.2543
#  gen: 27, stag: 1, mean: 47380.83, median: 15057.72, stdev: 55872.15, best: 1033.2543
#  gen: 28, stag: 2, mean: 49409.71, median: 9116.86, stdev: 55056.79, best: 1033.2543
#  gen: 29, stag: 3, mean: 48964.23, median: 6158.58, stdev: 57372.00, best: 1033.2543
#  gen: 30, stag: 4, mean: 36981.78, median: 10421.00, stdev: 45654.05, best: 1033.2543
#  gen: 31, stag: 5, mean: 41361.25, median: 33336.56, stdev: 40446.27, best: 1033.2543
#  gen: 32, stag: 6, mean: 38854.13, median: 13294.90, stdev: 44008.98, best: 1033.2543
#  gen: 33, stag: 7, mean: 40439.64, median: 15351.85, stdev: 43905.88, best: 1033.2543
#  gen: 34, stag: 8, mean: 44493.22, median: 35163.65, stdev: 43961.15, best: 1033.2543
#  gen: 35, stag: 9, mean: 47514.34, median: 48394.46, stdev: 42614.48, best: 1033.2543
#  gen: 36, stag: 10, mean: 52050.39, median: 27576.18, stdev: 50493.59, best: 1033.2543
#  gen: 37, stag: 11, mean: 40460.17, median: 8009.77, stdev: 47847.36, best: 1033.2543
#  gen: 38, stag: 12, mean: 59076.08, median: 42566.94, stdev: 59302.53, best: 1033.2543
#  gen: 39, stag: 0, mean: 58318.28, median: 35870.78, stdev: 64181.96, best: 739.2604
#  gen: 40, stag: 1, mean: 56816.45, median: 40110.42, stdev: 57315.84, best: 739.2604
#  gen: 41, stag: 2, mean: 66296.69, median: 64817.90, stdev: 62444.20, best: 739.2604
#  gen: 42, stag: 3, mean: 83315.76, median: 94068.66, stdev: 70133.65, best: 739.2604
#  gen: 43, stag: 4, mean: 85725.81, median: 120894.14, stdev: 73032.17, best: 739.2604
#  gen: 44, stag: 5, mean: 67132.87, median: 7574.59, stdev: 73063.69, best: 739.2604
#  gen: 45, stag: 6, mean: 45069.31, median: 33468.43, stdev: 49077.76, best: 739.2604
#  gen: 46, stag: 7, mean: 52923.18, median: 44744.29, stdev: 56418.85, best: 739.2604
#  gen: 47, stag: 8, mean: 52147.06, median: 18061.26, stdev: 64755.54, best: 739.2604
#  gen: 48, stag: 9, mean: 39589.47, median: 21606.65, stdev: 47715.90, best: 739.2604
#  gen: 49, stag: 10, mean: 28428.40, median: 25222.28, stdev: 27880.48, best: 739.2604
#  gen: 50, stag: 11, mean: 32154.03, median: 28710.22, stdev: 34130.07, best: 739.2604
#  gen: 51, stag: 12, mean: 22111.82, median: 14239.96, stdev: 22811.70, best: 739.2604
#  gen: 52, stag: 13, mean: 28893.79, median: 15425.52, stdev: 32669.83, best: 739.2604
#  gen: 53, stag: 14, mean: 23923.16, median: 12196.58, stdev: 28469.83, best: 739.2604
#  gen: 54, stag: 15, mean: 17120.77, median: 3585.61, stdev: 25594.66, best: 739.2604
#  gen: 55, stag: 16, mean: 16397.00, median: 4310.44, stdev: 20344.07, best: 739.2604
#  gen: 56, stag: 17, mean: 26737.43, median: 17378.87, stdev: 29313.13, best: 739.2604
#  gen: 57, stag: 18, mean: 34671.49, median: 22801.58, stdev: 34089.66, best: 739.2604
#  gen: 58, stag: 19, mean: 43552.42, median: 68228.03, stdev: 34384.88, best: 739.2604
#  gen: 59, stag: 20, mean: 42950.83, median: 68152.83, stdev: 35847.56, best: 739.2604
#  gen: 60, stag: 21, mean: 30719.45, median: 3127.04, stdev: 35743.16, best: 739.2604
#  gen: 61, stag: 22, mean: 30953.34, median: 14790.49, stdev: 32174.47, best: 739.2604
#  gen: 62, stag: 23, mean: 34764.61, median: 20819.23, stdev: 35416.65, best: 739.2604
#  gen: 63, stag: 24, mean: 45900.86, median: 41875.63, stdev: 42820.99, best: 739.2604
#  gen: 64, stag: 25, mean: 52969.83, median: 61554.97, stdev: 47561.53, best: 739.2604
#  gen: 65, stag: 26, mean: 41106.15, median: 19703.39, stdev: 42386.40, best: 739.2604
#  gen: 66, stag: 27, mean: 59170.07, median: 82609.55, stdev: 50727.65, best: 739.2604
#  gen: 67, stag: 28, mean: 38050.42, median: 3189.56, stdev: 48246.67, best: 739.2604
#  gen: 68, stag: 29, mean: 52545.97, median: 19317.71, stdev: 56085.96, best: 739.2604
#  gen: 69, stag: 30, mean: 45789.89, median: 20030.66, stdev: 50352.75, best: 739.2604
#  gen: 70, stag: 31, mean: 21221.11, median: 3612.41, stdev: 33195.90, best: 739.2604
#  gen: 71, stag: 32, mean: 25419.47, median: 24189.14, stdev: 24372.88, best: 739.2604
#  gen: 72, stag: 33, mean: 28901.86, median: 21242.76, stdev: 29768.99, best: 739.2604
#  gen: 73, stag: 34, mean: 31904.76, median: 21799.08, stdev: 35382.34, best: 739.2604
#  gen: 74, stag: 35, mean: 29074.17, median: 13726.15, stdev: 32947.77, best: 739.2604
#  gen: 75, stag: 36, mean: 27204.20, median: 21298.33, stdev: 27493.87, best: 739.2604
#  gen: 76, stag: 37, mean: 32637.61, median: 28734.87, stdev: 30470.33, best: 739.2604
#  gen: 77, stag: 38, mean: 28820.43, median: 11946.03, stdev: 31421.06, best: 739.2604
#  gen: 78, stag: 39, mean: 31858.72, median: 20483.56, stdev: 32663.00, best: 739.2604
#  gen: 79, stag: 40, mean: 19406.96, median: 13872.64, stdev: 21695.96, best: 739.2604
#  gen: 80, stag: 41, mean: 21653.34, median: 19152.88, stdev: 23710.61, best: 739.2604
#  gen: 81, stag: 42, mean: 27781.72, median: 31665.01, stdev: 25747.64, best: 739.2604
#  gen: 82, stag: 43, mean: 31750.73, median: 22114.94, stdev: 34029.52, best: 739.2604
#  gen: 83, stag: 44, mean: 32989.76, median: 33028.39, stdev: 30799.38, best: 739.2604
#  gen: 84, stag: 45, mean: 44404.34, median: 50805.30, stdev: 42227.75, best: 739.2604
#  gen: 85, stag: 46, mean: 41773.33, median: 5384.98, stdev: 54356.78, best: 739.2604
#  gen: 86, stag: 47, mean: 36784.34, median: 12292.77, stdev: 44827.81, best: 739.2604
#  gen: 87, stag: 48, mean: 39049.38, median: 26648.44, stdev: 41208.93, best: 739.2604
#  gen: 88, stag: 49, mean: 31041.15, median: 15700.50, stdev: 34134.89, best: 739.2604
#  gen: 89, stag: 50, mean: 28135.86, median: 21637.25, stdev: 30714.92, best: 739.2604
#  gen: 90, stag: 51, mean: 33757.56, median: 29214.20, stdev: 32293.34, best: 739.2604
#  gen: 91, stag: 52, mean: 24056.29, median: 2658.64, stdev: 30219.84, best: 739.2604
#  gen: 92, stag: 53, mean: 25446.23, median: 2900.70, stdev: 34809.87, best: 739.2604
#  gen: 93, stag: 54, mean: 31763.97, median: 15336.32, stdev: 42898.21, best: 739.2604
#  gen: 94, stag: 55, mean: 24237.79, median: 6650.82, stdev: 31948.05, best: 739.2604
#  gen: 95, stag: 56, mean: 42734.13, median: 30202.05, stdev: 45885.40, best: 739.2604
#  gen: 96, stag: 57, mean: 44946.71, median: 38390.51, stdev: 45355.52, best: 739.2604
#  gen: 97, stag: 58, mean: 36830.08, median: 22434.54, stdev: 37781.77, best: 739.2604
#  gen: 98, stag: 59, mean: 45653.43, median: 44175.13, stdev: 43761.54, best: 739.2604
#  gen: 99, stag: 60, mean: 45774.84, median: 32854.70, stdev: 43728.47, best: 739.2604
#  gen: 100, stag: 61, mean: 54598.07, median: 74263.53, stdev: 49469.52, best: 739.2604
#  gen: 101, stag: 62, mean: 64722.74, median: 84184.76, stdev: 61221.41, best: 739.2604
#  gen: 102, stag: 63, mean: 55547.11, median: 46340.72, stdev: 50805.70, best: 739.2604
#  gen: 103, stag: 64, mean: 54024.32, median: 53330.27, stdev: 47949.01, best: 739.2604
#  gen: 104, stag: 65, mean: 36684.72, median: 30124.21, stdev: 37577.24, best: 739.2604
#  gen: 105, stag: 66, mean: 27572.02, median: 7575.76, stdev: 34393.08, best: 739.2604
#  gen: 106, stag: 67, mean: 40751.45, median: 39388.79, stdev: 40982.73, best: 739.2604
#  gen: 107, stag: 68, mean: 36649.36, median: 32039.04, stdev: 37280.77, best: 739.2604
#  gen: 108, stag: 69, mean: 47284.34, median: 25769.09, stdev: 49440.24, best: 739.2604
#  gen: 109, stag: 70, mean: 58309.73, median: 61540.76, stdev: 56203.80, best: 739.2604
#  gen: 110, stag: 71, mean: 51492.37, median: 14763.03, stdev: 53865.25, best: 739.2604
#  gen: 111, stag: 72, mean: 32056.90, median: 2452.07, stdev: 46708.24, best: 739.2604
#  gen: 112, stag: 73, mean: 36750.32, median: 19634.69, stdev: 43110.81, best: 739.2604
#  gen: 113, stag: 74, mean: 32745.91, median: 22676.82, stdev: 36049.37, best: 739.2604
#  gen: 114, stag: 75, mean: 44987.06, median: 36034.11, stdev: 48524.90, best: 739.2604
#  gen: 115, stag: 76, mean: 51256.14, median: 42083.49, stdev: 55508.20, best: 739.2604
#  gen: 116, stag: 77, mean: 64517.71, median: 51417.84, stdev: 70046.06, best: 739.2604
#  gen: 117, stag: 78, mean: 45283.19, median: 3679.72, stdev: 60872.02, best: 739.2604
#  gen: 118, stag: 79, mean: 55005.08, median: 33300.67, stdev: 61458.14, best: 739.2604
#  gen: 119, stag: 80, mean: 63444.26, median: 51920.17, stdev: 68186.24, best: 739.2604
#  gen: 120, stag: 81, mean: 54291.15, median: 41184.77, stdev: 56750.34, best: 739.2604
#  gen: 121, stag: 82, mean: 45833.13, median: 6812.33, stdev: 56837.78, best: 739.2604
#  gen: 122, stag: 83, mean: 44599.80, median: 4763.47, stdev: 57910.91, best: 739.2604
#  gen: 123, stag: 84, mean: 55698.42, median: 9487.59, stdev: 74402.60, best: 739.2604
#  gen: 124, stag: 85, mean: 54199.31, median: 21974.32, stdev: 67851.85, best: 739.2604
#  gen: 125, stag: 86, mean: 61442.20, median: 38381.00, stdev: 63007.34, best: 739.2604
#  gen: 126, stag: 0, mean: 57239.16, median: 14894.16, stdev: 69413.82, best: 685.7089
#  gen: 127, stag: 1, mean: 33638.37, median: 2640.22, stdev: 54411.67, best: 685.7089
#  gen: 128, stag: 0, mean: 27157.88, median: 10314.49, stdev: 37807.95, best: 451.9790
#  gen: 129, stag: 1, mean: 31150.44, median: 19899.35, stdev: 35510.85, best: 451.9790
#  gen: 130, stag: 2, mean: 25510.46, median: 6344.86, stdev: 34250.76, best: 451.9790
#  gen: 131, stag: 3, mean: 30231.77, median: 9308.32, stdev: 37633.93, best: 451.9790
#  gen: 132, stag: 4, mean: 35517.90, median: 23463.85, stdev: 36913.89, best: 451.9790
#  gen: 133, stag: 5, mean: 45288.88, median: 38187.59, stdev: 45051.93, best: 451.9790
#  gen: 134, stag: 6, mean: 51444.26, median: 32589.87, stdev: 53177.64, best: 451.9790
#  gen: 135, stag: 7, mean: 44923.34, median: 9349.49, stdev: 48804.51, best: 451.9790
#  gen: 136, stag: 8, mean: 50364.11, median: 43141.95, stdev: 48906.84, best: 451.9790
#  gen: 137, stag: 9, mean: 42647.81, median: 3323.76, stdev: 55200.82, best: 451.9790
#  gen: 138, stag: 10, mean: 54370.76, median: 30661.53, stdev: 66192.87, best: 451.9790
#  gen: 139, stag: 11, mean: 44425.45, median: 14979.87, stdev: 57985.44, best: 451.9790
#  gen: 140, stag: 12, mean: 44331.10, median: 22090.98, stdev: 49167.04, best: 451.9790
#  gen: 141, stag: 13, mean: 42289.98, median: 17758.27, stdev: 49011.90, best: 451.9790
#  gen: 142, stag: 14, mean: 49862.24, median: 53149.51, stdev: 47410.91, best: 451.9790
#  gen: 143, stag: 15, mean: 66422.02, median: 81658.17, stdev: 58714.03, best: 451.9790
#  gen: 144, stag: 16, mean: 68468.73, median: 87104.49, stdev: 61026.48, best: 451.9790
#  gen: 145, stag: 17, mean: 81073.59, median: 105733.23, stdev: 68013.79, best: 451.9790
#  gen: 146, stag: 18, mean: 50818.08, median: 21048.17, stdev: 56982.83, best: 451.9790
#  gen: 147, stag: 19, mean: 55605.31, median: 37668.99, stdev: 56531.10, best: 451.9790
#  gen: 148, stag: 20, mean: 63936.07, median: 65087.27, stdev: 62716.69, best: 451.9790
#  gen: 149, stag: 21, mean: 75827.05, median: 100623.22, stdev: 66336.78, best: 451.9790
#  gen: 150, stag: 22, mean: 67803.39, median: 65312.41, stdev: 66078.34, best: 451.9790
#  gen: 151, stag: 23, mean: 65911.04, median: 36785.85, stdev: 72322.56, best: 451.9790
#  gen: 152, stag: 0, mean: 50352.65, median: 28456.44, stdev: 62852.65, best: 409.8593
#  gen: 153, stag: 1, mean: 45243.21, median: 36969.25, stdev: 48630.10, best: 409.8593
#  gen: 154, stag: 2, mean: 73562.65, median: 61788.31, stdev: 73408.57, best: 409.8593
#  gen: 155, stag: 3, mean: 78113.93, median: 79035.16, stdev: 75364.11, best: 409.8593
#  gen: 156, stag: 4, mean: 64513.14, median: 39590.58, stdev: 67807.64, best: 409.8593
#  gen: 157, stag: 5, mean: 65863.92, median: 64074.25, stdev: 66503.43, best: 409.8593
#  gen: 158, stag: 6, mean: 62497.53, median: 36656.54, stdev: 67908.30, best: 409.8593
#  gen: 159, stag: 7, mean: 91574.80, median: 109255.43, stdev: 82961.34, best: 409.8593
#  gen: 160, stag: 8, mean: 66759.30, median: 38582.29, stdev: 77941.67, best: 409.8593
#  gen: 161, stag: 9, mean: 53927.23, median: 10476.49, stdev: 67066.23, best: 409.8593
#  gen: 162, stag: 10, mean: 47440.43, median: 7644.82, stdev: 60021.80, best: 409.8593
#  gen: 163, stag: 11, mean: 53940.72, median: 43182.06, stdev: 57130.92, best: 409.8593
#  gen: 164, stag: 12, mean: 68230.92, median: 39765.68, stdev: 72971.45, best: 409.8593
#  gen: 165, stag: 13, mean: 71248.83, median: 62601.05, stdev: 72202.29, best: 409.8593
#  gen: 166, stag: 14, mean: 59689.29, median: 37383.67, stdev: 65569.68, best: 409.8593
#  gen: 167, stag: 15, mean: 39682.30, median: 26599.71, stdev: 45008.89, best: 409.8593
#  gen: 168, stag: 16, mean: 57604.56, median: 53910.87, stdev: 56989.88, best: 409.8593
#  gen: 169, stag: 17, mean: 72780.06, median: 92750.43, stdev: 64382.01, best: 409.8593
#  gen: 170, stag: 18, mean: 63305.61, median: 21240.61, stdev: 70590.02, best: 409.8593
#  gen: 171, stag: 19, mean: 73379.18, median: 67111.48, stdev: 72085.84, best: 409.8593
#  gen: 172, stag: 20, mean: 66422.06, median: 42721.90, stdev: 68840.97, best: 409.8593
#  gen: 173, stag: 21, mean: 61043.91, median: 28967.89, stdev: 66260.53, best: 409.8593
#  gen: 174, stag: 22, mean: 52809.75, median: 10664.87, stdev: 62424.53, best: 409.8593
#  gen: 175, stag: 23, mean: 59377.97, median: 14662.08, stdev: 69525.18, best: 409.8593
#  gen: 176, stag: 24, mean: 49933.34, median: 1797.43, stdev: 65736.04, best: 409.8593
#  gen: 177, stag: 25, mean: 58229.44, median: 23376.38, stdev: 67029.40, best: 409.8593
#  gen: 178, stag: 26, mean: 49553.23, median: 20773.26, stdev: 61049.75, best: 409.8593
#  gen: 179, stag: 27, mean: 48137.47, median: 1034.19, stdev: 57829.18, best: 409.8593
#  gen: 180, stag: 28, mean: 59565.81, median: 42387.57, stdev: 66748.40, best: 409.8593
#  gen: 181, stag: 29, mean: 37262.39, median: 10369.52, stdev: 54387.99, best: 409.8593
#  gen: 182, stag: 30, mean: 44152.55, median: 13322.87, stdev: 60951.98, best: 409.8593
#  gen: 183, stag: 31, mean: 52172.66, median: 9381.87, stdev: 67235.77, best: 409.8593
#  gen: 184, stag: 32, mean: 78126.51, median: 38767.41, stdev: 79426.91, best: 409.8593
#  gen: 185, stag: 33, mean: 73478.37, median: 18644.37, stdev: 78669.58, best: 409.8593
#  gen: 186, stag: 34, mean: 91778.79, median: 117574.07, stdev: 81137.90, best: 409.8593
#  gen: 187, stag: 35, mean: 66222.75, median: 35504.47, stdev: 73210.67, best: 409.8593
#  gen: 188, stag: 36, mean: 55468.46, median: 34239.78, stdev: 65924.81, best: 409.8593
#  gen: 189, stag: 37, mean: 36259.09, median: 10304.94, stdev: 45773.13, best: 409.8593
#  gen: 190, stag: 38, mean: 40286.22, median: 34241.45, stdev: 47104.08, best: 409.8593
#  gen: 191, stag: 39, mean: 33060.16, median: 9335.16, stdev: 42707.47, best: 409.8593
#  gen: 192, stag: 40, mean: 38009.76, median: 21038.84, stdev: 45820.41, best: 409.8593
#  gen: 193, stag: 41, mean: 16175.75, median: 4147.32, stdev: 21437.70, best: 409.8593
#  gen: 194, stag: 42, mean: 19457.19, median: 11331.61, stdev: 22507.04, best: 409.8593
#  gen: 195, stag: 43, mean: 17095.06, median: 11380.58, stdev: 18929.23, best: 409.8593
#  gen: 196, stag: 44, mean: 13958.57, median: 3652.27, stdev: 18912.46, best: 409.8593
#  gen: 197, stag: 45, mean: 21745.33, median: 14048.40, stdev: 30415.09, best: 409.8593
#  gen: 198, stag: 46, mean: 25835.15, median: 28443.66, stdev: 23928.86, best: 409.8593
#  gen: 199, stag: 47, mean: 26756.20, median: 25103.83, stdev: 26037.32, best: 409.8593
#  gen: 200, stag: 48, mean: 20756.07, median: 13406.80, stdev: 28449.69, best: 409.8593
#  gen: 201, stag: 49, mean: 14477.27, median: 2973.48, stdev: 25458.63, best: 409.8593
#  gen: 202, stag: 50, mean: 13208.22, median: 5446.96, stdev: 15274.76, best: 409.8593
#  gen: 203, stag: 51, mean: 12249.92, median: 8720.51, stdev: 11820.64, best: 409.8593
#  gen: 204, stag: 52, mean: 14067.44, median: 3036.21, stdev: 17587.18, best: 409.8593
#  gen: 205, stag: 53, mean: 14681.40, median: 4510.64, stdev: 24992.20, best: 409.8593
#  gen: 206, stag: 54, mean: 30036.43, median: 7408.47, stdev: 47026.18, best: 409.8593
#  gen: 207, stag: 55, mean: 35130.54, median: 18683.02, stdev: 51302.87, best: 409.8593
#  gen: 208, stag: 56, mean: 25534.15, median: 6217.16, stdev: 41818.50, best: 409.8593
#  gen: 209, stag: 57, mean: 24845.90, median: 11800.08, stdev: 28279.98, best: 409.8593
#  gen: 210, stag: 58, mean: 19193.14, median: 16037.21, stdev: 18423.30, best: 409.8593
#  gen: 211, stag: 59, mean: 18102.54, median: 10462.01, stdev: 21389.58, best: 409.8593
#  gen: 212, stag: 60, mean: 20559.06, median: 5703.72, stdev: 22713.66, best: 409.8593
#  gen: 213, stag: 61, mean: 25949.05, median: 9687.79, stdev: 29899.95, best: 409.8593
#  gen: 214, stag: 62, mean: 27796.20, median: 3453.28, stdev: 35004.60, best: 409.8593
#  gen: 215, stag: 63, mean: 21372.13, median: 9392.07, stdev: 25704.66, best: 409.8593
#  gen: 216, stag: 64, mean: 21693.73, median: 10509.72, stdev: 25037.13, best: 409.8593
#  gen: 217, stag: 65, mean: 33914.34, median: 39017.82, stdev: 31236.69, best: 409.8593
#  gen: 218, stag: 66, mean: 32187.36, median: 29967.48, stdev: 30568.18, best: 409.8593
#  gen: 219, stag: 67, mean: 29417.94, median: 9913.30, stdev: 33865.94, best: 409.8593
#  gen: 220, stag: 68, mean: 30779.09, median: 25500.88, stdev: 31277.73, best: 409.8593
#  gen: 221, stag: 69, mean: 32534.37, median: 33946.27, stdev: 31396.07, best: 409.8593
#  gen: 222, stag: 70, mean: 28604.73, median: 9394.62, stdev: 32305.40, best: 409.8593
#  gen: 223, stag: 71, mean: 28424.93, median: 15050.66, stdev: 31821.44, best: 409.8593
#  gen: 224, stag: 72, mean: 22204.83, median: 4074.95, stdev: 27936.24, best: 409.8593
#  gen: 225, stag: 73, mean: 22579.94, median: 5202.55, stdev: 30974.02, best: 409.8593
#  gen: 226, stag: 74, mean: 26394.39, median: 11045.02, stdev: 30561.19, best: 409.8593
#  gen: 227, stag: 75, mean: 33918.16, median: 28703.96, stdev: 35031.86, best: 409.8593
#  gen: 228, stag: 76, mean: 44950.86, median: 32747.30, stdev: 45142.28, best: 409.8593
#  gen: 229, stag: 77, mean: 57496.68, median: 55829.72, stdev: 54834.36, best: 409.8593
#  gen: 230, stag: 78, mean: 48773.08, median: 28728.15, stdev: 50854.21, best: 409.8593
#  gen: 231, stag: 79, mean: 32116.63, median: 1160.54, stdev: 43294.31, best: 409.8593
#  gen: 232, stag: 80, mean: 34014.51, median: 16388.90, stdev: 42270.89, best: 409.8593
#  gen: 233, stag: 81, mean: 26982.83, median: 15804.65, stdev: 31625.26, best: 409.8593
#  gen: 234, stag: 82, mean: 26983.10, median: 10097.96, stdev: 34318.18, best: 409.8593
#  gen: 235, stag: 83, mean: 25841.36, median: 21951.25, stdev: 31542.08, best: 409.8593
#  gen: 236, stag: 84, mean: 21896.62, median: 5856.14, stdev: 25667.43, best: 409.8593
#  gen: 237, stag: 85, mean: 15235.92, median: 4448.13, stdev: 18952.80, best: 409.8593
#  gen: 238, stag: 86, mean: 21630.58, median: 9170.86, stdev: 23663.90, best: 409.8593
#  gen: 239, stag: 87, mean: 19048.09, median: 5412.30, stdev: 26258.12, best: 409.8593
#  gen: 240, stag: 88, mean: 19745.84, median: 5326.74, stdev: 23640.62, best: 409.8593
#  gen: 241, stag: 89, mean: 20965.71, median: 8639.86, stdev: 25101.72, best: 409.8593
#  gen: 242, stag: 90, mean: 29605.26, median: 27579.28, stdev: 32179.10, best: 409.8593
#  gen: 243, stag: 91, mean: 31374.70, median: 12184.00, stdev: 40980.57, best: 409.8593
#  gen: 244, stag: 92, mean: 28247.53, median: 6053.86, stdev: 36928.78, best: 409.8593
#  gen: 245, stag: 93, mean: 27397.32, median: 5685.44, stdev: 35960.77, best: 409.8593
#  gen: 246, stag: 94, mean: 21827.31, median: 5323.61, stdev: 31968.53, best: 409.8593
#  gen: 247, stag: 95, mean: 22781.33, median: 13066.60, stdev: 31094.55, best: 409.8593
#  gen: 248, stag: 96, mean: 13869.51, median: 9587.60, stdev: 18502.85, best: 409.8593
#  gen: 249, stag: 97, mean: 25052.33, median: 14940.46, stdev: 26948.92, best: 409.8593
#  gen: 250, stag: 98, mean: 21762.88, median: 9089.43, stdev: 29494.84, best: 409.8593
#  gen: 251, stag: 99, mean: 30435.87, median: 13544.38, stdev: 34097.48, best: 409.8593
#  gen: 252, stag: 100, mean: 24288.67, median: 889.61, stdev: 34628.67, best: 409.8593
#  gen: 253, stag: 101, mean: 26343.47, median: 9753.84, stdev: 33644.92, best: 409.8593
#  gen: 254, stag: 102, mean: 26167.03, median: 10112.01, stdev: 31528.87, best: 409.8593
#  gen: 255, stag: 103, mean: 44371.19, median: 36989.51, stdev: 44706.46, best: 409.8593
#  gen: 256, stag: 104, mean: 44869.00, median: 21080.46, stdev: 48063.95, best: 409.8593
#  gen: 257, stag: 105, mean: 21251.45, median: 2938.97, stdev: 34731.75, best: 409.8593
#  gen: 258, stag: 106, mean: 27283.70, median: 10028.60, stdev: 31416.04, best: 409.8593
#  gen: 259, stag: 107, mean: 28276.76, median: 14233.47, stdev: 35017.22, best: 409.8593
#  gen: 260, stag: 108, mean: 25054.14, median: 4304.03, stdev: 33895.55, best: 409.8593
#  gen: 261, stag: 109, mean: 19604.58, median: 815.47, stdev: 36502.71, best: 409.8593
#  gen: 262, stag: 110, mean: 19464.32, median: 1804.32, stdev: 30902.72, best: 409.8593
#  gen: 263, stag: 111, mean: 19385.75, median: 2033.71, stdev: 28683.73, best: 409.8593
#  gen: 264, stag: 112, mean: 13636.31, median: 1438.75, stdev: 25466.08, best: 409.8593
#  gen: 265, stag: 113, mean: 15672.04, median: 2738.25, stdev: 22632.82, best: 409.8593
#  gen: 266, stag: 114, mean: 13476.82, median: 2408.53, stdev: 20169.32, best: 409.8593
#  gen: 267, stag: 115, mean: 20710.12, median: 9695.06, stdev: 29229.53, best: 409.8593
#  gen: 268, stag: 116, mean: 30087.94, median: 9945.97, stdev: 37848.05, best: 409.8593
#  gen: 269, stag: 117, mean: 30005.57, median: 1497.20, stdev: 39822.98, best: 409.8593
#  gen: 270, stag: 118, mean: 38760.28, median: 19208.65, stdev: 46249.40, best: 409.8593
#  gen: 271, stag: 119, mean: 38084.73, median: 24939.18, stdev: 40775.93, best: 409.8593
#  gen: 272, stag: 120, mean: 52338.33, median: 59912.49, stdev: 48811.12, best: 409.8593
#  gen: 273, stag: 121, mean: 61411.49, median: 67360.23, stdev: 55831.45, best: 409.8593
#  gen: 274, stag: 122, mean: 65085.90, median: 63588.12, stdev: 63350.65, best: 409.8593
#  gen: 275, stag: 123, mean: 67076.37, median: 53169.98, stdev: 69128.59, best: 409.8593
#  gen: 276, stag: 124, mean: 83333.01, median: 42996.45, stdev: 85433.12, best: 409.8593
#  gen: 277, stag: 125, mean: 89163.99, median: 53533.95, stdev: 92942.70, best: 409.8593
#  gen: 278, stag: 126, mean: 91298.67, median: 54256.85, stdev: 94627.14, best: 409.8593
#  gen: 279, stag: 127, mean: 72329.79, median: 39611.61, stdev: 84648.68, best: 409.8593
#  gen: 280, stag: 128, mean: 85915.95, median: 47717.18, stdev: 92829.24, best: 409.8593
#  gen: 281, stag: 129, mean: 57863.28, median: 15632.03, stdev: 73456.95, best: 409.8593
#  gen: 282, stag: 130, mean: 44935.13, median: 7160.15, stdev: 59134.54, best: 409.8593
#  gen: 283, stag: 131, mean: 65290.22, median: 37987.93, stdev: 72181.17, best: 409.8593
#  gen: 284, stag: 132, mean: 79939.25, median: 39632.60, stdev: 86264.62, best: 409.8593
#  gen: 285, stag: 133, mean: 55582.23, median: 4835.03, stdev: 72479.64, best: 409.8593
#  gen: 286, stag: 134, mean: 41945.37, median: 8587.92, stdev: 60160.10, best: 409.8593
#  gen: 287, stag: 0, mean: 30894.14, median: 846.32, stdev: 56419.70, best: 325.7950
#  gen: 288, stag: 1, mean: 7808.18, median: 775.49, stdev: 20201.62, best: 325.7950
#  gen: 289, stag: 2, mean: 8996.38, median: 1045.76, stdev: 15544.22, best: 325.7950
#  gen: 290, stag: 3, mean: 13703.57, median: 2229.21, stdev: 17687.53, best: 325.7950
#  gen: 291, stag: 4, mean: 11853.62, median: 2323.90, stdev: 16301.71, best: 325.7950
#  gen: 292, stag: 5, mean: 13325.94, median: 2853.83, stdev: 18108.56, best: 325.7950
#  gen: 293, stag: 6, mean: 11810.14, median: 1724.48, stdev: 18500.21, best: 325.7950
#  gen: 294, stag: 7, mean: 17438.60, median: 3363.40, stdev: 28346.79, best: 325.7950
#  gen: 295, stag: 8, mean: 20758.93, median: 10231.57, stdev: 22498.30, best: 325.7950
#  gen: 296, stag: 9, mean: 21053.32, median: 7558.80, stdev: 23070.95, best: 325.7950
#  gen: 297, stag: 10, mean: 14916.55, median: 3276.75, stdev: 17357.55, best: 325.7950
#  gen: 298, stag: 11, mean: 11238.34, median: 2251.29, stdev: 14855.24, best: 325.7950
#  gen: 299, stag: 12, mean: 12720.72, median: 3151.69, stdev: 15630.02, best: 325.7950
#  gen: 300, stag: 13, mean: 16173.53, median: 7818.00, stdev: 18849.43, best: 325.7950
#  gen: 301, stag: 14, mean: 11890.79, median: 2747.96, stdev: 16502.67, best: 325.7950

#  Total generations: 301
#  Total CPU time: 6.640532493 s
#  Best score: 325.79497700832974
#  Real chromosome of best solution: [-0.32033338 -0.79760675 -0.23605274 -0.60206898 -0.05011025 -0.07962769
#  -0.97535394]
#  Binary chromosome of best solution: []
