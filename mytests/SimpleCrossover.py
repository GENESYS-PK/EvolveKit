from evolvekit import *
from evolvekit.operators.Ga.real.mutation.UniformMutation import UniformMutation
from evolvekit.operators.Ga.real.crossover.SimpleCrossover import (
    SimpleCrossover,
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
crossover_real = SimpleCrossover()
crossover_bin = OnePointCrossoverBin()
mutation_real = UniformMutation()
mutation_bin = VirusInfectionMutationBin(
    [
        [0, 1, "*", "*", 0, 1, "*", 0, 1, "*", "*", 0, 1, "*", 0, 1],
        [1, "*", 0, "*", 1, "*", 0, "*", 1, "*", 0, "*", 1, "*", 0, "*"],
    ]
)

island.set_elite_count(20)
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
#  gen: 2, stag: 0, mean: 74944.64, median: 66149.58, stdev: 47192.73, best: 6655.9551
#  gen: 3, stag: 1, mean: 68730.58, median: 85365.47, stdev: 38885.42, best: 6655.9551
#  gen: 4, stag: 2, mean: 66548.36, median: 76340.57, stdev: 36746.51, best: 6655.9551
#  gen: 5, stag: 3, mean: 69737.62, median: 75369.69, stdev: 42317.77, best: 6655.9551
#  gen: 6, stag: 4, mean: 87044.59, median: 114871.69, stdev: 43095.52, best: 6655.9551
#  gen: 7, stag: 0, mean: 78780.80, median: 90715.39, stdev: 43872.53, best: 4743.5562
#  gen: 8, stag: 1, mean: 79864.81, median: 100693.66, stdev: 44108.67, best: 4743.5562
#  gen: 9, stag: 2, mean: 82517.87, median: 105601.21, stdev: 43108.99, best: 4743.5562
#  gen: 10, stag: 3, mean: 88311.77, median: 114024.40, stdev: 41858.40, best: 4743.5562
#  gen: 11, stag: 4, mean: 95026.65, median: 114903.29, stdev: 42207.83, best: 4743.5562
#  gen: 12, stag: 5, mean: 95588.63, median: 116300.59, stdev: 42478.12, best: 4743.5562
#  gen: 13, stag: 6, mean: 96483.35, median: 117037.32, stdev: 42981.39, best: 4743.5562
#  gen: 14, stag: 7, mean: 75103.82, median: 115774.27, stdev: 49264.17, best: 4743.5562
#  gen: 15, stag: 8, mean: 84290.49, median: 116867.94, stdev: 45409.76, best: 4743.5562
#  gen: 16, stag: 9, mean: 80352.38, median: 117278.02, stdev: 48353.44, best: 4743.5562
#  gen: 17, stag: 10, mean: 72477.26, median: 92887.08, stdev: 51720.35, best: 4743.5562
#  gen: 18, stag: 11, mean: 49431.94, median: 18111.27, stdev: 45585.47, best: 4743.5562
#  gen: 19, stag: 12, mean: 75750.59, median: 108082.76, stdev: 48727.35, best: 4743.5562
#  gen: 20, stag: 13, mean: 81756.41, median: 119288.99, stdev: 49847.31, best: 4743.5562
#  gen: 21, stag: 14, mean: 74271.76, median: 117559.57, stdev: 52395.16, best: 4743.5562
#  gen: 22, stag: 15, mean: 81267.77, median: 116309.96, stdev: 45022.85, best: 4743.5562
#  gen: 23, stag: 16, mean: 82001.99, median: 119285.96, stdev: 50136.31, best: 4743.5562
#  gen: 24, stag: 17, mean: 82224.23, median: 119128.88, stdev: 48016.77, best: 4743.5562
#  gen: 25, stag: 18, mean: 77583.49, median: 102554.71, stdev: 47763.86, best: 4743.5562
#  gen: 26, stag: 19, mean: 88437.21, median: 119453.87, stdev: 48788.12, best: 4743.5562
#  gen: 27, stag: 20, mean: 97188.49, median: 119467.99, stdev: 44598.97, best: 4743.5562
#  gen: 28, stag: 21, mean: 97215.29, median: 119503.44, stdev: 44612.37, best: 4743.5562
#  gen: 29, stag: 22, mean: 97272.27, median: 119559.03, stdev: 44641.46, best: 4743.5562
#  gen: 30, stag: 23, mean: 97244.75, median: 119546.98, stdev: 44627.09, best: 4743.5562
#  gen: 31, stag: 24, mean: 97231.26, median: 119547.43, stdev: 44620.80, best: 4743.5562
#  gen: 32, stag: 25, mean: 97335.25, median: 119583.56, stdev: 44677.51, best: 4743.5562
#  gen: 33, stag: 26, mean: 81969.70, median: 119589.53, stdev: 51796.95, best: 4743.5562
#  gen: 34, stag: 27, mean: 96607.47, median: 119593.29, stdev: 45148.55, best: 4743.5562
#  gen: 35, stag: 28, mean: 90644.17, median: 119596.73, stdev: 48582.37, best: 4743.5562
#  gen: 36, stag: 29, mean: 97050.57, median: 119602.35, stdev: 45331.64, best: 4743.5562
#  gen: 37, stag: 30, mean: 96962.76, median: 119602.48, stdev: 45279.15, best: 4743.5562
#  gen: 38, stag: 31, mean: 97157.12, median: 119603.14, stdev: 45416.97, best: 4743.5562
#  gen: 39, stag: 32, mean: 98700.33, median: 119603.15, stdev: 46407.18, best: 4743.5562
#  gen: 40, stag: 33, mean: 100302.48, median: 119603.28, stdev: 47364.60, best: 4743.5562
#  gen: 41, stag: 0, mean: 87129.40, median: 119603.27, stdev: 52951.91, best: 631.0091
#  gen: 42, stag: 1, mean: 82847.77, median: 120088.27, stdev: 51461.52, best: 631.0091
#  gen: 43, stag: 2, mean: 99599.31, median: 121839.33, stdev: 47549.15, best: 631.0091
#  gen: 44, stag: 3, mean: 99501.63, median: 121973.82, stdev: 47470.96, best: 631.0091
#  gen: 45, stag: 4, mean: 101051.87, median: 124077.32, stdev: 48249.66, best: 631.0091
#  gen: 46, stag: 5, mean: 80639.75, median: 125724.75, stdev: 56003.70, best: 631.0091
#  gen: 47, stag: 6, mean: 102311.27, median: 125894.81, stdev: 48978.27, best: 631.0091
#  gen: 48, stag: 7, mean: 71296.51, median: 80074.16, stdev: 57187.17, best: 631.0091
#  gen: 49, stag: 8, mean: 76346.05, median: 127217.60, stdev: 56122.20, best: 631.0091
#  gen: 50, stag: 9, mean: 95294.34, median: 128836.07, stdev: 53601.17, best: 631.0091
#  gen: 51, stag: 10, mean: 92352.03, median: 128774.59, stdev: 53305.40, best: 631.0091
#  gen: 52, stag: 11, mean: 88097.74, median: 128876.31, stdev: 56972.54, best: 631.0091
#  gen: 53, stag: 12, mean: 78399.97, median: 128869.68, stdev: 57146.26, best: 631.0091
#  gen: 54, stag: 13, mean: 97846.49, median: 128947.93, stdev: 50840.90, best: 631.0091
#  gen: 55, stag: 14, mean: 99810.91, median: 128951.74, stdev: 49906.04, best: 631.0091
#  gen: 56, stag: 15, mean: 103977.68, median: 128985.13, stdev: 50043.33, best: 631.0091
#  gen: 57, stag: 16, mean: 103982.85, median: 129002.08, stdev: 50045.90, best: 631.0091
#  gen: 58, stag: 17, mean: 103987.77, median: 129008.74, stdev: 50048.36, best: 631.0091
#  gen: 59, stag: 18, mean: 103993.51, median: 129014.28, stdev: 50051.23, best: 631.0091
#  gen: 60, stag: 19, mean: 103640.46, median: 129017.73, stdev: 49999.16, best: 631.0091
#  gen: 61, stag: 20, mean: 100734.95, median: 129018.09, stdev: 49405.53, best: 631.0091
#  gen: 62, stag: 21, mean: 103996.99, median: 129020.84, stdev: 50052.97, best: 631.0091
#  gen: 63, stag: 22, mean: 103968.25, median: 129019.50, stdev: 50064.67, best: 631.0091
#  gen: 64, stag: 23, mean: 95055.87, median: 129016.18, stdev: 54702.67, best: 631.0091
#  gen: 65, stag: 24, mean: 93078.57, median: 129020.24, stdev: 54162.76, best: 631.0091
#  gen: 66, stag: 25, mean: 103898.35, median: 129022.10, stdev: 50252.40, best: 631.0091
#  gen: 67, stag: 26, mean: 93248.80, median: 129022.07, stdev: 55690.85, best: 631.0091
#  gen: 68, stag: 27, mean: 93687.55, median: 129022.09, stdev: 55493.28, best: 631.0091
#  gen: 69, stag: 28, mean: 103830.51, median: 129022.94, stdev: 50375.39, best: 631.0091
#  gen: 70, stag: 29, mean: 103834.72, median: 129023.38, stdev: 50377.49, best: 631.0091
#  gen: 71, stag: 30, mean: 76527.00, median: 129020.57, stdev: 58692.91, best: 631.0091
#  gen: 72, stag: 31, mean: 87102.85, median: 129021.91, stdev: 56735.70, best: 631.0091
#  gen: 73, stag: 32, mean: 90760.05, median: 129006.21, stdev: 52064.64, best: 631.0091
#  gen: 74, stag: 33, mean: 78137.68, median: 87571.52, stdev: 50429.00, best: 631.0091
#  gen: 75, stag: 34, mean: 90770.21, median: 108827.48, stdev: 47064.66, best: 631.0091
#  gen: 76, stag: 35, mean: 91267.50, median: 109692.32, stdev: 45810.79, best: 631.0091
#  gen: 77, stag: 36, mean: 99449.72, median: 124107.68, stdev: 48548.15, best: 631.0091
#  gen: 78, stag: 37, mean: 98817.04, median: 124698.62, stdev: 50924.25, best: 631.0091
#  gen: 79, stag: 38, mean: 101512.59, median: 125527.03, stdev: 49325.91, best: 631.0091
#  gen: 80, stag: 39, mean: 102212.66, median: 126451.31, stdev: 49672.29, best: 631.0091
#  gen: 81, stag: 40, mean: 103342.93, median: 128493.95, stdev: 50219.75, best: 631.0091
#  gen: 82, stag: 41, mean: 94154.20, median: 128528.65, stdev: 53499.41, best: 631.0091
#  gen: 83, stag: 42, mean: 94954.55, median: 128533.19, stdev: 54479.56, best: 631.0091
#  gen: 84, stag: 43, mean: 103634.06, median: 128448.65, stdev: 50483.64, best: 631.0091
#  gen: 85, stag: 44, mean: 103365.42, median: 128434.19, stdev: 50233.19, best: 631.0091
#  gen: 86, stag: 45, mean: 103389.27, median: 128491.60, stdev: 50245.08, best: 631.0091
#  gen: 87, stag: 46, mean: 103289.45, median: 128506.43, stdev: 50206.78, best: 631.0091
#  gen: 88, stag: 47, mean: 103433.99, median: 128554.82, stdev: 50267.39, best: 631.0091
#  gen: 89, stag: 48, mean: 87642.46, median: 128570.82, stdev: 56998.78, best: 631.0091
#  gen: 90, stag: 49, mean: 92817.78, median: 128526.20, stdev: 55820.38, best: 631.0091
#  gen: 91, stag: 50, mean: 103160.69, median: 128594.31, stdev: 50340.32, best: 631.0091
#  gen: 92, stag: 51, mean: 103051.71, median: 128549.76, stdev: 50340.69, best: 631.0091
#  gen: 93, stag: 52, mean: 92139.18, median: 128568.54, stdev: 56222.47, best: 631.0091
#  gen: 94, stag: 53, mean: 103369.48, median: 128588.64, stdev: 50428.13, best: 631.0091
#  gen: 95, stag: 54, mean: 103363.51, median: 128573.83, stdev: 50425.15, best: 631.0091
#  gen: 96, stag: 55, mean: 60011.03, median: 34567.95, stdev: 58027.44, best: 631.0091
#  gen: 97, stag: 56, mean: 68335.84, median: 62911.71, stdev: 57212.92, best: 631.0091
#  gen: 98, stag: 57, mean: 85881.27, median: 127725.73, stdev: 56151.70, best: 631.0091
#  gen: 99, stag: 58, mean: 98902.92, median: 127562.73, stdev: 49139.19, best: 631.0091
#  gen: 100, stag: 59, mean: 72075.27, median: 102439.47, stdev: 56010.86, best: 631.0091
#  gen: 101, stag: 60, mean: 55715.71, median: 17575.87, stdev: 57945.80, best: 631.0091
#  gen: 102, stag: 61, mean: 87527.91, median: 128573.19, stdev: 54601.75, best: 631.0091
#  gen: 103, stag: 62, mean: 103146.93, median: 128578.76, stdev: 50701.60, best: 631.0091
#  gen: 104, stag: 63, mean: 91550.05, median: 128579.18, stdev: 57471.56, best: 631.0091
#  gen: 105, stag: 64, mean: 103264.32, median: 128581.61, stdev: 50893.97, best: 631.0091
#  gen: 106, stag: 65, mean: 93364.22, median: 128582.51, stdev: 55597.90, best: 631.0091
#  gen: 107, stag: 66, mean: 94965.10, median: 128582.58, stdev: 55305.01, best: 631.0091
#  gen: 108, stag: 67, mean: 79104.68, median: 126214.36, stdev: 60159.02, best: 631.0091
#  gen: 109, stag: 68, mean: 106012.12, median: 128582.89, stdev: 54070.28, best: 631.0091
#  gen: 110, stag: 69, mean: 95745.67, median: 128582.92, stdev: 54483.73, best: 631.0091
#  gen: 111, stag: 70, mean: 89980.34, median: 128582.96, stdev: 54464.26, best: 631.0091
#  gen: 112, stag: 71, mean: 98261.51, median: 128582.98, stdev: 50170.29, best: 631.0091
#  gen: 113, stag: 72, mean: 103135.68, median: 128582.99, stdev: 50894.71, best: 631.0091
#  gen: 114, stag: 73, mean: 102788.03, median: 128582.98, stdev: 50838.40, best: 631.0091
#  gen: 115, stag: 74, mean: 103135.69, median: 128583.01, stdev: 50894.71, best: 631.0091
#  gen: 116, stag: 75, mean: 103141.75, median: 128583.02, stdev: 50897.78, best: 631.0091
#  gen: 117, stag: 76, mean: 103135.70, median: 128583.02, stdev: 50894.72, best: 631.0091
#  gen: 118, stag: 77, mean: 103214.16, median: 128583.02, stdev: 50939.91, best: 631.0091
#  gen: 119, stag: 78, mean: 95120.81, median: 128583.02, stdev: 55207.10, best: 631.0091
#  gen: 120, stag: 79, mean: 85908.19, median: 128582.47, stdev: 55523.03, best: 631.0091
#  gen: 121, stag: 80, mean: 88812.61, median: 107122.04, stdev: 47706.64, best: 631.0091
#  gen: 122, stag: 81, mean: 102799.04, median: 128583.02, stdev: 50943.03, best: 631.0091
#  gen: 123, stag: 82, mean: 79840.95, median: 128579.07, stdev: 58276.88, best: 631.0091
#  gen: 124, stag: 83, mean: 75282.37, median: 113063.12, stdev: 55639.08, best: 631.0091
#  gen: 125, stag: 84, mean: 90153.89, median: 117256.84, stdev: 49852.13, best: 631.0091
#  gen: 126, stag: 85, mean: 91139.56, median: 122236.62, stdev: 53305.14, best: 631.0091
#  gen: 127, stag: 86, mean: 79031.82, median: 113329.56, stdev: 53013.30, best: 631.0091
#  gen: 128, stag: 87, mean: 92091.14, median: 123928.93, stdev: 51804.18, best: 631.0091
#  gen: 129, stag: 88, mean: 90064.18, median: 125536.50, stdev: 54674.84, best: 631.0091
#  gen: 130, stag: 89, mean: 87562.02, median: 124456.95, stdev: 54808.57, best: 631.0091
#  gen: 131, stag: 90, mean: 91039.03, median: 126731.14, stdev: 54474.01, best: 631.0091
#  gen: 132, stag: 91, mean: 78980.61, median: 125628.34, stdev: 55691.98, best: 631.0091
#  gen: 133, stag: 92, mean: 90627.90, median: 127746.94, stdev: 53163.49, best: 631.0091
#  gen: 134, stag: 93, mean: 78391.54, median: 100884.47, stdev: 54107.81, best: 631.0091
#  gen: 135, stag: 94, mean: 65146.72, median: 56954.36, stdev: 57586.79, best: 631.0091
#  gen: 136, stag: 95, mean: 100095.07, median: 125771.35, stdev: 49618.61, best: 631.0091
#  gen: 137, stag: 96, mean: 101252.67, median: 125803.05, stdev: 50044.24, best: 631.0091
#  gen: 138, stag: 97, mean: 90874.72, median: 125731.39, stdev: 54559.26, best: 631.0091
#  gen: 139, stag: 98, mean: 101895.46, median: 127049.29, stdev: 50346.02, best: 631.0091
#  gen: 140, stag: 99, mean: 102526.67, median: 127897.82, stdev: 50660.12, best: 631.0091
#  gen: 141, stag: 100, mean: 102745.22, median: 128050.84, stdev: 50765.18, best: 631.0091
#  gen: 142, stag: 101, mean: 91060.92, median: 128055.60, stdev: 56073.12, best: 631.0091
#  gen: 143, stag: 102, mean: 86329.62, median: 128068.80, stdev: 53561.82, best: 631.0091
#  gen: 144, stag: 103, mean: 81030.97, median: 128047.44, stdev: 56431.48, best: 631.0091
#  gen: 145, stag: 104, mean: 88692.62, median: 128135.29, stdev: 53707.27, best: 631.0091
#  gen: 146, stag: 105, mean: 93006.13, median: 128140.91, stdev: 51254.78, best: 631.0091
#  gen: 147, stag: 106, mean: 98040.29, median: 128190.83, stdev: 50349.68, best: 631.0091
#  gen: 148, stag: 107, mean: 89795.09, median: 103098.49, stdev: 47715.47, best: 631.0091
#  gen: 149, stag: 108, mean: 102813.75, median: 128202.46, stdev: 50806.01, best: 631.0091
#  gen: 150, stag: 109, mean: 102829.13, median: 128233.27, stdev: 50813.70, best: 631.0091
#  gen: 151, stag: 110, mean: 102762.83, median: 128250.57, stdev: 50786.01, best: 631.0091
#  gen: 152, stag: 111, mean: 102844.64, median: 128256.16, stdev: 50821.45, best: 631.0091
#  gen: 153, stag: 112, mean: 102848.74, median: 128259.44, stdev: 50823.50, best: 631.0091
#  gen: 154, stag: 113, mean: 102763.77, median: 128260.88, stdev: 50788.18, best: 631.0091
#  gen: 155, stag: 114, mean: 102850.82, median: 128262.23, stdev: 50824.54, best: 631.0091
#  gen: 156, stag: 115, mean: 102850.68, median: 128262.64, stdev: 50824.47, best: 631.0091
#  gen: 157, stag: 116, mean: 102860.07, median: 128262.96, stdev: 50829.29, best: 631.0091
#  gen: 158, stag: 117, mean: 85563.38, median: 128262.74, stdev: 57583.02, best: 631.0091
#  gen: 159, stag: 118, mean: 102569.90, median: 128263.13, stdev: 50697.43, best: 631.0091
#  gen: 160, stag: 119, mean: 92098.82, median: 128263.15, stdev: 56693.92, best: 631.0091
#  gen: 161, stag: 120, mean: 70224.97, median: 113715.41, stdev: 60411.19, best: 631.0091
#  gen: 162, stag: 121, mean: 92456.74, median: 128263.91, stdev: 54297.64, best: 631.0091
#  gen: 163, stag: 122, mean: 88954.37, median: 128263.92, stdev: 56442.09, best: 631.0091
#  gen: 164, stag: 123, mean: 102477.08, median: 128263.97, stdev: 50726.72, best: 631.0091
#  gen: 165, stag: 124, mean: 86253.15, median: 127967.97, stdev: 57142.62, best: 631.0091
#  gen: 166, stag: 125, mean: 83781.94, median: 127547.16, stdev: 58802.86, best: 631.0091
#  gen: 167, stag: 126, mean: 103621.04, median: 127824.51, stdev: 52066.24, best: 631.0091
#  gen: 168, stag: 127, mean: 102507.77, median: 128008.62, stdev: 50679.59, best: 631.0091
#  gen: 169, stag: 0, mean: 76468.16, median: 127654.48, stdev: 59325.17, best: 560.4496
#  gen: 170, stag: 1, mean: 62254.87, median: 67459.11, stdev: 56760.19, best: 560.4496
#  gen: 171, stag: 2, mean: 54966.30, median: 28908.91, stdev: 55125.13, best: 560.4496
#  gen: 172, stag: 3, mean: 71984.48, median: 91020.03, stdev: 51522.28, best: 560.4496
#  gen: 173, stag: 4, mean: 79779.58, median: 108879.88, stdev: 52579.57, best: 560.4496
#  gen: 174, stag: 5, mean: 96845.58, median: 128007.04, stdev: 49621.30, best: 560.4496
#  gen: 175, stag: 6, mean: 90900.45, median: 127902.22, stdev: 55290.01, best: 560.4496
#  gen: 176, stag: 7, mean: 102827.28, median: 128066.86, stdev: 50924.63, best: 560.4496
#  gen: 177, stag: 8, mean: 102655.08, median: 128070.00, stdev: 50811.83, best: 560.4496
#  gen: 178, stag: 9, mean: 102169.00, median: 128072.39, stdev: 50721.53, best: 560.4496
#  gen: 179, stag: 10, mean: 102442.92, median: 128072.88, stdev: 50753.66, best: 560.4496
#  gen: 180, stag: 11, mean: 103510.23, median: 128073.72, stdev: 51740.57, best: 560.4496
#  gen: 181, stag: 12, mean: 103344.42, median: 128073.74, stdev: 51369.60, best: 560.4496
#  gen: 182, stag: 13, mean: 104292.49, median: 128074.57, stdev: 52358.34, best: 560.4496
#  gen: 183, stag: 14, mean: 92648.18, median: 120269.52, stdev: 58120.97, best: 560.4496
#  gen: 184, stag: 15, mean: 68100.56, median: 63268.91, stdev: 61288.68, best: 560.4496
#  gen: 185, stag: 16, mean: 87508.46, median: 120443.16, stdev: 58305.33, best: 560.4496
#  gen: 186, stag: 17, mean: 90161.59, median: 127543.39, stdev: 62571.43, best: 560.4496
#  gen: 187, stag: 18, mean: 110264.06, median: 137804.90, stdev: 55085.96, best: 560.4496
#  gen: 188, stag: 19, mean: 108943.35, median: 131534.23, stdev: 54358.98, best: 560.4496
#  gen: 189, stag: 20, mean: 111224.34, median: 141345.89, stdev: 55341.53, best: 560.4496
#  gen: 190, stag: 21, mean: 112138.53, median: 141508.86, stdev: 55937.93, best: 560.4496
#  gen: 191, stag: 22, mean: 104528.10, median: 141434.96, stdev: 60488.37, best: 560.4496
#  gen: 192, stag: 23, mean: 113756.58, median: 141561.86, stdev: 56383.14, best: 560.4496
#  gen: 193, stag: 24, mean: 113962.90, median: 142021.66, stdev: 56486.09, best: 560.4496
#  gen: 194, stag: 25, mean: 113469.84, median: 142478.42, stdev: 56459.46, best: 560.4496
#  gen: 195, stag: 26, mean: 113455.09, median: 142607.02, stdev: 56615.25, best: 560.4496
#  gen: 196, stag: 27, mean: 115646.90, median: 142622.68, stdev: 58591.65, best: 560.4496
#  gen: 197, stag: 28, mean: 114413.79, median: 142634.34, stdev: 56712.22, best: 560.4496
#  gen: 198, stag: 29, mean: 114705.08, median: 142706.26, stdev: 56934.38, best: 560.4496
#  gen: 199, stag: 30, mean: 108027.97, median: 142738.24, stdev: 60429.45, best: 560.4496
#  gen: 200, stag: 31, mean: 104947.46, median: 142668.97, stdev: 60178.76, best: 560.4496
#  gen: 201, stag: 32, mean: 100997.97, median: 142697.88, stdev: 61544.36, best: 560.4496
#  gen: 202, stag: 33, mean: 85730.38, median: 142674.37, stdev: 65590.38, best: 560.4496
#  gen: 203, stag: 34, mean: 100133.97, median: 142730.09, stdev: 59626.35, best: 560.4496
#  gen: 204, stag: 35, mean: 109624.14, median: 142835.19, stdev: 55830.73, best: 560.4496
#  gen: 205, stag: 36, mean: 114352.16, median: 142842.44, stdev: 56751.76, best: 560.4496
#  gen: 206, stag: 37, mean: 114466.01, median: 142842.53, stdev: 56757.38, best: 560.4496
#  gen: 207, stag: 38, mean: 114472.96, median: 142845.36, stdev: 56760.86, best: 560.4496
#  gen: 208, stag: 39, mean: 114475.66, median: 142856.01, stdev: 56762.20, best: 560.4496
#  gen: 209, stag: 40, mean: 114185.72, median: 142864.70, stdev: 56676.93, best: 560.4496
#  gen: 210, stag: 41, mean: 105617.94, median: 142864.26, stdev: 61196.86, best: 560.4496
#  gen: 211, stag: 42, mean: 114472.75, median: 142864.52, stdev: 56780.09, best: 560.4496
#  gen: 212, stag: 43, mean: 108389.17, median: 142864.78, stdev: 60284.93, best: 560.4496
#  gen: 213, stag: 44, mean: 95909.79, median: 142864.79, stdev: 64668.68, best: 560.4496
#  gen: 214, stag: 45, mean: 106242.65, median: 142864.81, stdev: 57669.94, best: 560.4496
#  gen: 215, stag: 46, mean: 114055.65, median: 142864.81, stdev: 56724.39, best: 560.4496
#  gen: 216, stag: 47, mean: 114474.39, median: 142864.82, stdev: 56780.93, best: 560.4496
#  gen: 217, stag: 48, mean: 102052.63, median: 142864.82, stdev: 62183.50, best: 560.4496
#  gen: 218, stag: 49, mean: 104080.50, median: 142864.82, stdev: 63871.49, best: 560.4496
#  gen: 219, stag: 50, mean: 122399.12, median: 142864.82, stdev: 66674.11, best: 560.4496
#  gen: 220, stag: 51, mean: 121593.66, median: 142864.82, stdev: 65683.30, best: 560.4496
#  gen: 221, stag: 52, mean: 124904.64, median: 142864.82, stdev: 68329.48, best: 560.4496
#  gen: 222, stag: 53, mean: 124981.74, median: 142864.82, stdev: 68061.15, best: 560.4496
#  gen: 223, stag: 54, mean: 127143.54, median: 142864.85, stdev: 69059.27, best: 560.4496
#  gen: 224, stag: 55, mean: 130683.45, median: 149204.51, stdev: 69952.25, best: 560.4496
#  gen: 225, stag: 56, mean: 140448.59, median: 171058.24, stdev: 73948.10, best: 560.4496
#  gen: 226, stag: 57, mean: 135090.38, median: 171058.26, stdev: 67641.13, best: 560.4496
#  gen: 227, stag: 58, mean: 131724.08, median: 171058.26, stdev: 71073.73, best: 560.4496
#  gen: 228, stag: 59, mean: 122916.55, median: 170978.61, stdev: 73884.84, best: 560.4496
#  gen: 229, stag: 60, mean: 125562.46, median: 171063.45, stdev: 75645.59, best: 560.4496
#  gen: 230, stag: 61, mean: 116318.23, median: 171067.31, stdev: 77457.77, best: 560.4496
#  gen: 231, stag: 62, mean: 148123.46, median: 172775.55, stdev: 74342.92, best: 560.4496
#  gen: 232, stag: 63, mean: 153046.49, median: 194853.91, stdev: 76372.14, best: 560.4496
#  gen: 233, stag: 64, mean: 146239.42, median: 194856.07, stdev: 83927.66, best: 560.4496
#  gen: 234, stag: 65, mean: 135157.11, median: 194853.25, stdev: 84638.65, best: 560.4496
#  gen: 235, stag: 66, mean: 138067.27, median: 194856.99, stdev: 86160.95, best: 560.4496
#  gen: 236, stag: 67, mean: 156072.61, median: 194882.11, stdev: 77614.73, best: 560.4496
#  gen: 237, stag: 68, mean: 155714.92, median: 194890.66, stdev: 77522.16, best: 560.4496
#  gen: 238, stag: 69, mean: 126221.71, median: 194897.49, stdev: 90077.75, best: 560.4496
#  gen: 239, stag: 70, mean: 140710.89, median: 194897.79, stdev: 84673.45, best: 560.4496
#  gen: 240, stag: 71, mean: 142299.49, median: 194897.81, stdev: 85798.90, best: 560.4496
#  gen: 241, stag: 72, mean: 137678.14, median: 194898.37, stdev: 83713.80, best: 560.4496
#  gen: 242, stag: 73, mean: 151113.77, median: 194899.45, stdev: 77007.09, best: 560.4496
#  gen: 243, stag: 74, mean: 156086.23, median: 194899.20, stdev: 77626.52, best: 560.4496
#  gen: 244, stag: 75, mean: 140973.29, median: 194899.36, stdev: 85646.14, best: 560.4496
#  gen: 245, stag: 76, mean: 137615.78, median: 194899.57, stdev: 86117.67, best: 560.4496
#  gen: 246, stag: 77, mean: 156064.55, median: 194899.80, stdev: 77622.67, best: 560.4496
#  gen: 247, stag: 78, mean: 156083.84, median: 194899.78, stdev: 77632.07, best: 560.4496
#  gen: 248, stag: 79, mean: 138670.96, median: 194899.91, stdev: 86404.76, best: 560.4496
#  gen: 249, stag: 80, mean: 155571.89, median: 194899.98, stdev: 77533.07, best: 560.4496
#  gen: 250, stag: 81, mean: 142137.17, median: 194899.98, stdev: 84238.33, best: 560.4496
#  gen: 251, stag: 82, mean: 155587.38, median: 194900.00, stdev: 77543.37, best: 560.4496
#  gen: 252, stag: 83, mean: 137940.60, median: 194899.98, stdev: 86095.46, best: 560.4496
#  gen: 253, stag: 84, mean: 156081.76, median: 194900.03, stdev: 77634.22, best: 560.4496
#  gen: 254, stag: 85, mean: 137068.99, median: 194900.04, stdev: 85331.64, best: 560.4496
#  gen: 255, stag: 86, mean: 111850.17, median: 186004.44, stdev: 91121.32, best: 560.4496
#  gen: 256, stag: 87, mean: 84149.89, median: 78083.95, stdev: 79591.83, best: 560.4496
#  gen: 257, stag: 88, mean: 114089.32, median: 123028.69, stdev: 80115.28, best: 560.4496
#  gen: 258, stag: 89, mean: 145840.60, median: 194899.11, stdev: 80679.84, best: 560.4496
#  gen: 259, stag: 90, mean: 155623.86, median: 194899.56, stdev: 77513.28, best: 560.4496
#  gen: 260, stag: 91, mean: 145660.07, median: 194899.22, stdev: 82361.57, best: 560.4496
#  gen: 261, stag: 92, mean: 139035.93, median: 194899.20, stdev: 78683.73, best: 560.4496
#  gen: 262, stag: 93, mean: 146915.23, median: 194899.47, stdev: 76249.28, best: 560.4496
#  gen: 263, stag: 94, mean: 145539.71, median: 194899.67, stdev: 82112.40, best: 560.4496
#  gen: 264, stag: 95, mean: 156074.90, median: 194899.84, stdev: 77649.90, best: 560.4496
#  gen: 265, stag: 96, mean: 156072.46, median: 194899.90, stdev: 77648.69, best: 560.4496
#  gen: 266, stag: 97, mean: 131295.38, median: 194899.89, stdev: 87927.80, best: 560.4496
#  gen: 267, stag: 98, mean: 127688.08, median: 194899.90, stdev: 92636.28, best: 560.4496
#  gen: 268, stag: 99, mean: 123890.16, median: 194899.90, stdev: 89617.39, best: 560.4496
#  gen: 269, stag: 100, mean: 114395.28, median: 193439.83, stdev: 90664.72, best: 560.4496
#  gen: 270, stag: 101, mean: 153213.73, median: 194899.91, stdev: 76575.42, best: 560.4496
#  gen: 271, stag: 102, mean: 156060.89, median: 194899.91, stdev: 77678.05, best: 560.4496
#  gen: 272, stag: 103, mean: 155548.36, median: 194899.91, stdev: 77589.13, best: 560.4496
#  gen: 273, stag: 104, mean: 110575.22, median: 185606.35, stdev: 91840.84, best: 560.4496
#  gen: 274, stag: 105, mean: 142175.53, median: 194899.91, stdev: 83477.65, best: 560.4496
#  gen: 275, stag: 106, mean: 135759.16, median: 194899.91, stdev: 80341.81, best: 560.4496
#  gen: 276, stag: 107, mean: 140852.13, median: 194899.91, stdev: 76721.99, best: 560.4496
#  gen: 277, stag: 108, mean: 118611.76, median: 166730.89, stdev: 83823.22, best: 560.4496
#  gen: 278, stag: 109, mean: 115997.77, median: 169262.04, stdev: 86045.92, best: 560.4496
#  gen: 279, stag: 110, mean: 129204.40, median: 169866.12, stdev: 84176.25, best: 560.4496
#  gen: 280, stag: 111, mean: 133891.09, median: 163841.15, stdev: 70983.26, best: 560.4496
#  gen: 281, stag: 112, mean: 134353.38, median: 166760.73, stdev: 72800.17, best: 560.4496
#  gen: 282, stag: 113, mean: 130183.60, median: 191773.04, stdev: 84689.65, best: 560.4496
#  gen: 283, stag: 114, mean: 111577.11, median: 133085.67, stdev: 84613.96, best: 560.4496
#  gen: 284, stag: 115, mean: 123492.25, median: 138376.49, stdev: 77056.68, best: 560.4496
#  gen: 285, stag: 0, mean: 138786.46, median: 194746.93, stdev: 83226.85, best: 559.9865
#  gen: 286, stag: 1, mean: 155553.93, median: 194780.72, stdev: 77485.85, best: 559.9865
#  gen: 287, stag: 2, mean: 155041.64, median: 194803.04, stdev: 77486.70, best: 559.9865
#  gen: 288, stag: 3, mean: 155981.92, median: 194821.68, stdev: 77683.28, best: 559.9865
#  gen: 289, stag: 0, mean: 145040.58, median: 194827.12, stdev: 83708.66, best: 552.9370
#  gen: 290, stag: 1, mean: 137762.04, median: 194833.45, stdev: 86293.55, best: 552.9370
#  gen: 291, stag: 2, mean: 126608.96, median: 194834.28, stdev: 89778.44, best: 552.9370
#  gen: 292, stag: 3, mean: 155999.84, median: 194848.49, stdev: 77696.27, best: 552.9370
#  gen: 293, stag: 4, mean: 155777.34, median: 194849.90, stdev: 77617.11, best: 552.9370
#  gen: 294, stag: 5, mean: 145503.68, median: 194849.69, stdev: 82888.99, best: 552.9370
#  gen: 295, stag: 6, mean: 141048.07, median: 194848.73, stdev: 86030.98, best: 552.9370
#  gen: 296, stag: 7, mean: 155999.41, median: 194849.99, stdev: 77702.53, best: 552.9370
#  gen: 297, stag: 8, mean: 155999.97, median: 194850.68, stdev: 77702.81, best: 552.9370
#  gen: 298, stag: 9, mean: 156060.49, median: 194851.21, stdev: 77735.36, best: 552.9370
#  gen: 299, stag: 10, mean: 155344.80, median: 194852.88, stdev: 77649.61, best: 552.9370
#  gen: 300, stag: 11, mean: 145259.85, median: 194853.37, stdev: 83433.94, best: 552.9370
#  gen: 301, stag: 12, mean: 156964.26, median: 194853.42, stdev: 78793.97, best: 552.9370

#  Total generations: 301
#  Total CPU time: 5.079259986 s
#  Best score: 552.93701072694
#  Real chromosome of best solution: [-0.51186496  1.65189366  1.02763376  0.94883183  1.377525    1.42174593
#   1.56554883]
#  Binary chromosome of best solution: []
