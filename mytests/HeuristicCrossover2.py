from evolvekit import *
from evolvekit.operators.Ga.real.mutation.UniformMutation import UniformMutation
from evolvekit.operators.Ga.real.crossover.HeuristicCrossover2 import (
    HeuristicCrossover2,
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
crossover_real = HeuristicCrossover2()
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
#  gen: 2, stag: 1, mean: 84324.14, median: 69223.99, stdev: 58285.13, best: 6656.6039
#  gen: 3, stag: 0, mean: 79971.94, median: 69107.72, stdev: 59904.42, best: 1017.1588
#  gen: 4, stag: 1, mean: 58470.79, median: 53206.95, stdev: 46031.04, best: 1017.1588
#  gen: 5, stag: 2, mean: 62877.22, median: 47053.77, stdev: 53143.80, best: 1017.1588
#  gen: 6, stag: 3, mean: 59375.83, median: 40686.26, stdev: 52033.58, best: 1017.1588
#  gen: 7, stag: 4, mean: 63331.64, median: 48791.79, stdev: 57595.19, best: 1017.1588
#  gen: 8, stag: 5, mean: 69632.84, median: 61417.31, stdev: 56756.40, best: 1017.1588
#  gen: 9, stag: 6, mean: 67704.00, median: 49079.65, stdev: 71153.35, best: 1017.1588
#  gen: 10, stag: 7, mean: 55700.69, median: 34541.84, stdev: 58859.60, best: 1017.1588
#  gen: 11, stag: 8, mean: 53912.51, median: 49898.30, stdev: 45149.17, best: 1017.1588
#  gen: 12, stag: 0, mean: 51599.06, median: 41909.15, stdev: 46095.33, best: 1012.1173
#  gen: 13, stag: 1, mean: 55453.58, median: 46021.65, stdev: 49426.70, best: 1012.1173
#  gen: 14, stag: 2, mean: 44917.83, median: 36495.51, stdev: 42583.19, best: 1012.1173
#  gen: 15, stag: 3, mean: 71613.62, median: 58939.54, stdev: 65743.14, best: 1012.1173
#  gen: 16, stag: 4, mean: 64417.17, median: 44265.80, stdev: 57526.29, best: 1012.1173
#  gen: 17, stag: 5, mean: 75004.85, median: 63399.34, stdev: 65356.03, best: 1012.1173
#  gen: 18, stag: 6, mean: 70640.48, median: 63208.74, stdev: 55666.30, best: 1012.1173
#  gen: 19, stag: 7, mean: 79546.57, median: 80682.20, stdev: 61119.48, best: 1012.1173
#  gen: 20, stag: 8, mean: 78893.47, median: 83498.25, stdev: 66706.93, best: 1012.1173
#  gen: 21, stag: 9, mean: 92884.15, median: 90944.61, stdev: 68932.52, best: 1012.1173
#  gen: 22, stag: 10, mean: 57020.22, median: 44658.55, stdev: 59108.54, best: 1012.1173
#  gen: 23, stag: 11, mean: 47871.96, median: 34499.67, stdev: 51265.54, best: 1012.1173
#  gen: 24, stag: 12, mean: 66142.42, median: 52665.37, stdev: 63246.87, best: 1012.1173
#  gen: 25, stag: 13, mean: 61335.45, median: 52971.98, stdev: 63085.52, best: 1012.1173
#  gen: 26, stag: 14, mean: 80588.83, median: 73049.30, stdev: 64884.75, best: 1012.1173
#  gen: 27, stag: 15, mean: 85153.08, median: 66069.13, stdev: 75342.96, best: 1012.1173
#  gen: 28, stag: 16, mean: 93847.38, median: 97188.35, stdev: 71047.07, best: 1012.1173
#  gen: 29, stag: 17, mean: 83546.42, median: 80232.25, stdev: 69310.37, best: 1012.1173
#  gen: 30, stag: 18, mean: 64383.27, median: 38284.82, stdev: 63916.36, best: 1012.1173
#  gen: 31, stag: 19, mean: 75673.70, median: 57891.33, stdev: 60425.21, best: 1012.1173
#  gen: 32, stag: 20, mean: 89523.31, median: 60852.01, stdev: 84823.27, best: 1012.1173
#  gen: 33, stag: 21, mean: 110296.88, median: 105814.98, stdev: 84261.23, best: 1012.1173
#  gen: 34, stag: 22, mean: 69621.89, median: 50348.91, stdev: 65714.33, best: 1012.1173
#  gen: 35, stag: 23, mean: 48593.54, median: 32617.53, stdev: 52464.32, best: 1012.1173
#  gen: 36, stag: 24, mean: 74470.40, median: 53515.44, stdev: 66413.56, best: 1012.1173
#  gen: 37, stag: 25, mean: 67926.67, median: 56448.71, stdev: 59409.08, best: 1012.1173
#  gen: 38, stag: 26, mean: 58968.76, median: 50623.58, stdev: 49375.79, best: 1012.1173
#  gen: 39, stag: 27, mean: 67436.28, median: 64429.53, stdev: 51273.92, best: 1012.1173
#  gen: 40, stag: 28, mean: 75235.43, median: 65961.20, stdev: 68644.72, best: 1012.1173
#  gen: 41, stag: 29, mean: 73303.05, median: 69651.27, stdev: 59773.28, best: 1012.1173
#  gen: 42, stag: 30, mean: 78575.64, median: 65312.02, stdev: 68612.70, best: 1012.1173
#  gen: 43, stag: 31, mean: 62432.12, median: 56795.37, stdev: 59475.49, best: 1012.1173
#  gen: 44, stag: 32, mean: 72031.65, median: 74339.14, stdev: 66937.42, best: 1012.1173
#  gen: 45, stag: 33, mean: 62837.50, median: 71985.33, stdev: 50207.19, best: 1012.1173
#  gen: 46, stag: 34, mean: 61140.72, median: 44844.63, stdev: 57414.69, best: 1012.1173
#  gen: 47, stag: 35, mean: 60912.47, median: 38999.61, stdev: 59181.65, best: 1012.1173
#  gen: 48, stag: 36, mean: 63764.97, median: 55587.71, stdev: 52877.78, best: 1012.1173
#  gen: 49, stag: 37, mean: 52739.36, median: 46119.48, stdev: 49375.46, best: 1012.1173
#  gen: 50, stag: 38, mean: 67262.26, median: 63043.78, stdev: 54442.40, best: 1012.1173
#  gen: 51, stag: 39, mean: 64810.78, median: 56074.93, stdev: 52507.25, best: 1012.1173
#  gen: 52, stag: 40, mean: 68413.19, median: 65109.33, stdev: 59615.61, best: 1012.1173
#  gen: 53, stag: 41, mean: 88601.92, median: 87626.82, stdev: 64280.47, best: 1012.1173
#  gen: 54, stag: 42, mean: 103418.52, median: 97271.04, stdev: 78320.86, best: 1012.1173
#  gen: 55, stag: 43, mean: 67427.68, median: 53589.87, stdev: 58424.86, best: 1012.1173
#  gen: 56, stag: 44, mean: 74288.62, median: 77903.63, stdev: 64728.79, best: 1012.1173
#  gen: 57, stag: 45, mean: 45376.72, median: 30271.21, stdev: 48917.02, best: 1012.1173
#  gen: 58, stag: 46, mean: 81002.07, median: 82601.34, stdev: 59251.52, best: 1012.1173
#  gen: 59, stag: 47, mean: 99570.74, median: 92786.13, stdev: 78726.62, best: 1012.1173
#  gen: 60, stag: 48, mean: 92064.00, median: 86514.07, stdev: 77429.79, best: 1012.1173
#  gen: 61, stag: 49, mean: 85740.18, median: 84380.81, stdev: 64709.83, best: 1012.1173
#  gen: 62, stag: 50, mean: 74788.93, median: 65337.56, stdev: 64223.19, best: 1012.1173
#  gen: 63, stag: 51, mean: 66895.62, median: 47197.52, stdev: 60784.90, best: 1012.1173
#  gen: 64, stag: 52, mean: 70315.67, median: 69561.73, stdev: 59234.64, best: 1012.1173
#  gen: 65, stag: 53, mean: 62383.50, median: 44445.96, stdev: 64952.13, best: 1012.1173
#  gen: 66, stag: 54, mean: 73570.90, median: 58325.92, stdev: 71700.04, best: 1012.1173
#  gen: 67, stag: 55, mean: 51285.51, median: 27471.21, stdev: 53777.89, best: 1012.1173
#  gen: 68, stag: 56, mean: 62400.78, median: 59409.47, stdev: 54250.75, best: 1012.1173
#  gen: 69, stag: 57, mean: 61161.82, median: 44264.32, stdev: 53635.88, best: 1012.1173
#  gen: 70, stag: 58, mean: 61314.32, median: 55070.01, stdev: 54520.28, best: 1012.1173
#  gen: 71, stag: 59, mean: 72751.78, median: 47715.31, stdev: 71815.13, best: 1012.1173
#  gen: 72, stag: 60, mean: 60322.50, median: 43105.64, stdev: 61627.06, best: 1012.1173
#  gen: 73, stag: 61, mean: 77444.02, median: 74135.38, stdev: 62852.89, best: 1012.1173
#  gen: 74, stag: 62, mean: 69622.08, median: 79042.95, stdev: 57060.73, best: 1012.1173
#  gen: 75, stag: 63, mean: 101283.35, median: 112397.58, stdev: 77012.75, best: 1012.1173
#  gen: 76, stag: 64, mean: 65213.72, median: 44578.77, stdev: 69168.98, best: 1012.1173
#  gen: 77, stag: 65, mean: 63225.70, median: 54071.03, stdev: 60738.41, best: 1012.1173
#  gen: 78, stag: 66, mean: 88197.47, median: 76331.75, stdev: 69191.59, best: 1012.1173
#  gen: 79, stag: 67, mean: 87270.44, median: 85661.80, stdev: 66379.31, best: 1012.1173
#  gen: 80, stag: 68, mean: 85111.50, median: 80852.40, stdev: 70754.84, best: 1012.1173
#  gen: 81, stag: 69, mean: 84162.31, median: 83081.92, stdev: 65092.90, best: 1012.1173
#  gen: 82, stag: 70, mean: 61377.34, median: 54110.13, stdev: 58514.62, best: 1012.1173
#  gen: 83, stag: 71, mean: 55107.18, median: 49096.41, stdev: 48591.46, best: 1012.1173
#  gen: 84, stag: 72, mean: 59032.91, median: 53277.57, stdev: 50652.17, best: 1012.1173
#  gen: 85, stag: 73, mean: 67394.60, median: 61928.44, stdev: 56508.30, best: 1012.1173
#  gen: 86, stag: 74, mean: 59325.90, median: 51459.37, stdev: 50202.89, best: 1012.1173
#  gen: 87, stag: 75, mean: 74610.35, median: 71651.32, stdev: 54723.79, best: 1012.1173
#  gen: 88, stag: 76, mean: 79654.96, median: 79507.74, stdev: 51565.50, best: 1012.1173
#  gen: 89, stag: 0, mean: 63317.81, median: 66040.40, stdev: 54679.45, best: 375.7073
#  gen: 90, stag: 1, mean: 67091.54, median: 76180.31, stdev: 42609.51, best: 375.7073
#  gen: 91, stag: 2, mean: 80775.38, median: 92740.89, stdev: 50403.20, best: 375.7073
#  gen: 92, stag: 3, mean: 88994.78, median: 96236.48, stdev: 57103.88, best: 375.7073
#  gen: 93, stag: 4, mean: 77155.03, median: 91141.97, stdev: 57128.49, best: 375.7073
#  gen: 94, stag: 5, mean: 94676.18, median: 100695.18, stdev: 66777.81, best: 375.7073
#  gen: 95, stag: 6, mean: 89194.31, median: 97655.84, stdev: 51684.16, best: 375.7073
#  gen: 96, stag: 7, mean: 94433.88, median: 100871.53, stdev: 58326.45, best: 375.7073
#  gen: 97, stag: 8, mean: 81340.96, median: 89609.60, stdev: 67844.17, best: 375.7073
#  gen: 98, stag: 9, mean: 84321.56, median: 91374.19, stdev: 52982.63, best: 375.7073
#  gen: 99, stag: 10, mean: 87381.99, median: 80084.13, stdev: 58104.24, best: 375.7073
#  gen: 100, stag: 11, mean: 85964.36, median: 89306.16, stdev: 60426.93, best: 375.7073
#  gen: 101, stag: 12, mean: 71582.58, median: 67513.05, stdev: 62252.14, best: 375.7073
#  gen: 102, stag: 13, mean: 59342.47, median: 40249.35, stdev: 61454.95, best: 375.7073
#  gen: 103, stag: 14, mean: 55873.25, median: 49631.10, stdev: 47626.25, best: 375.7073
#  gen: 104, stag: 15, mean: 87341.30, median: 74253.05, stdev: 71118.79, best: 375.7073
#  gen: 105, stag: 16, mean: 56762.43, median: 42453.55, stdev: 55132.22, best: 375.7073
#  gen: 106, stag: 17, mean: 60838.94, median: 44998.00, stdev: 55716.58, best: 375.7073
#  gen: 107, stag: 18, mean: 52500.89, median: 40245.78, stdev: 45823.81, best: 375.7073
#  gen: 108, stag: 19, mean: 54732.95, median: 45153.77, stdev: 50902.63, best: 375.7073
#  gen: 109, stag: 20, mean: 53645.00, median: 38712.13, stdev: 51650.32, best: 375.7073
#  gen: 110, stag: 21, mean: 64862.70, median: 53990.92, stdev: 54052.50, best: 375.7073
#  gen: 111, stag: 22, mean: 69058.25, median: 63757.57, stdev: 61976.19, best: 375.7073
#  gen: 112, stag: 23, mean: 72885.80, median: 59395.84, stdev: 61213.55, best: 375.7073
#  gen: 113, stag: 0, mean: 68074.50, median: 59142.83, stdev: 62042.53, best: 304.3603
#  gen: 114, stag: 1, mean: 55979.93, median: 42229.33, stdev: 58336.89, best: 304.3603
#  gen: 115, stag: 2, mean: 61164.72, median: 51866.45, stdev: 54788.59, best: 304.3603
#  gen: 116, stag: 3, mean: 69028.29, median: 60016.87, stdev: 58171.81, best: 304.3603
#  gen: 117, stag: 4, mean: 66452.29, median: 41055.74, stdev: 67429.19, best: 304.3603
#  gen: 118, stag: 5, mean: 59273.70, median: 36358.78, stdev: 53206.64, best: 304.3603
#  gen: 119, stag: 6, mean: 57797.63, median: 44124.49, stdev: 50801.07, best: 304.3603
#  gen: 120, stag: 7, mean: 41713.41, median: 21768.96, stdev: 49636.88, best: 304.3603
#  gen: 121, stag: 8, mean: 50292.54, median: 35678.57, stdev: 46583.22, best: 304.3603
#  gen: 122, stag: 9, mean: 62650.68, median: 60197.71, stdev: 50207.94, best: 304.3603
#  gen: 123, stag: 10, mean: 50130.79, median: 25085.10, stdev: 55742.96, best: 304.3603
#  gen: 124, stag: 11, mean: 59173.23, median: 33009.99, stdev: 62556.90, best: 304.3603
#  gen: 125, stag: 12, mean: 67564.91, median: 49676.16, stdev: 56741.80, best: 304.3603
#  gen: 126, stag: 13, mean: 72931.02, median: 67151.68, stdev: 62054.33, best: 304.3603
#  gen: 127, stag: 14, mean: 76773.71, median: 62187.72, stdev: 60077.32, best: 304.3603
#  gen: 128, stag: 15, mean: 61512.71, median: 59195.41, stdev: 50483.32, best: 304.3603
#  gen: 129, stag: 16, mean: 79565.61, median: 71060.70, stdev: 66351.19, best: 304.3603
#  gen: 130, stag: 17, mean: 89228.98, median: 85867.25, stdev: 64393.27, best: 304.3603
#  gen: 131, stag: 18, mean: 78304.20, median: 81514.72, stdev: 58294.12, best: 304.3603
#  gen: 132, stag: 19, mean: 72976.51, median: 58711.09, stdev: 62571.96, best: 304.3603
#  gen: 133, stag: 20, mean: 62716.88, median: 41207.92, stdev: 77218.11, best: 304.3603
#  gen: 134, stag: 21, mean: 66172.54, median: 56129.23, stdev: 61045.11, best: 304.3603
#  gen: 135, stag: 22, mean: 62373.07, median: 41590.42, stdev: 60084.99, best: 304.3603
#  gen: 136, stag: 23, mean: 54510.75, median: 40974.41, stdev: 51936.13, best: 304.3603
#  gen: 137, stag: 24, mean: 71793.53, median: 50106.45, stdev: 65176.31, best: 304.3603
#  gen: 138, stag: 25, mean: 74593.45, median: 56816.20, stdev: 69578.04, best: 304.3603
#  gen: 139, stag: 26, mean: 68959.92, median: 63393.92, stdev: 54726.17, best: 304.3603
#  gen: 140, stag: 27, mean: 62361.97, median: 58495.86, stdev: 59229.50, best: 304.3603
#  gen: 141, stag: 28, mean: 77736.13, median: 66706.93, stdev: 73017.39, best: 304.3603
#  gen: 142, stag: 29, mean: 99949.87, median: 94824.11, stdev: 79114.95, best: 304.3603
#  gen: 143, stag: 30, mean: 98629.98, median: 95351.99, stdev: 73504.90, best: 304.3603
#  gen: 144, stag: 31, mean: 68800.50, median: 65006.74, stdev: 67521.23, best: 304.3603
#  gen: 145, stag: 32, mean: 70506.50, median: 60621.23, stdev: 67187.94, best: 304.3603
#  gen: 146, stag: 33, mean: 84409.38, median: 90155.51, stdev: 59739.29, best: 304.3603
#  gen: 147, stag: 34, mean: 76843.73, median: 82749.49, stdev: 53580.72, best: 304.3603
#  gen: 148, stag: 35, mean: 76467.45, median: 82955.94, stdev: 54798.26, best: 304.3603
#  gen: 149, stag: 36, mean: 82180.73, median: 70963.98, stdev: 68146.90, best: 304.3603
#  gen: 150, stag: 37, mean: 67487.16, median: 47272.55, stdev: 68959.78, best: 304.3603
#  gen: 151, stag: 38, mean: 77395.35, median: 83695.41, stdev: 56910.01, best: 304.3603
#  gen: 152, stag: 39, mean: 79915.10, median: 79621.23, stdev: 54442.17, best: 304.3603
#  gen: 153, stag: 40, mean: 58322.36, median: 54648.93, stdev: 52857.53, best: 304.3603
#  gen: 154, stag: 41, mean: 71842.77, median: 69173.78, stdev: 63693.02, best: 304.3603
#  gen: 155, stag: 42, mean: 76571.26, median: 75169.04, stdev: 58771.55, best: 304.3603
#  gen: 156, stag: 43, mean: 54001.80, median: 42740.00, stdev: 53273.13, best: 304.3603
#  gen: 157, stag: 44, mean: 68119.40, median: 52112.79, stdev: 60396.68, best: 304.3603
#  gen: 158, stag: 45, mean: 58567.03, median: 48926.07, stdev: 48198.65, best: 304.3603
#  gen: 159, stag: 46, mean: 58970.89, median: 51554.01, stdev: 54109.61, best: 304.3603
#  gen: 160, stag: 47, mean: 90739.24, median: 87446.68, stdev: 72476.26, best: 304.3603
#  gen: 161, stag: 48, mean: 99666.81, median: 108470.62, stdev: 79826.29, best: 304.3603
#  gen: 162, stag: 49, mean: 81568.06, median: 66086.92, stdev: 75071.70, best: 304.3603
#  gen: 163, stag: 50, mean: 57399.06, median: 35549.63, stdev: 59269.98, best: 304.3603
#  gen: 164, stag: 51, mean: 53578.86, median: 36145.69, stdev: 54684.00, best: 304.3603
#  gen: 165, stag: 52, mean: 63013.80, median: 51903.61, stdev: 57995.77, best: 304.3603
#  gen: 166, stag: 53, mean: 78040.66, median: 70551.42, stdev: 65255.91, best: 304.3603
#  gen: 167, stag: 54, mean: 105526.86, median: 114082.60, stdev: 75692.33, best: 304.3603
#  gen: 168, stag: 55, mean: 95386.89, median: 81983.97, stdev: 75218.55, best: 304.3603
#  gen: 169, stag: 56, mean: 124110.07, median: 134276.37, stdev: 84442.85, best: 304.3603
#  gen: 170, stag: 57, mean: 95826.49, median: 81267.00, stdev: 77568.13, best: 304.3603
#  gen: 171, stag: 58, mean: 100627.03, median: 91666.99, stdev: 84328.23, best: 304.3603
#  gen: 172, stag: 59, mean: 125335.56, median: 113940.52, stdev: 91258.33, best: 304.3603
#  gen: 173, stag: 60, mean: 74592.09, median: 58075.23, stdev: 75421.88, best: 304.3603
#  gen: 174, stag: 61, mean: 87273.14, median: 83201.06, stdev: 73062.81, best: 304.3603
#  gen: 175, stag: 62, mean: 72675.10, median: 46137.58, stdev: 71109.74, best: 304.3603
#  gen: 176, stag: 63, mean: 79874.93, median: 55503.33, stdev: 75861.60, best: 304.3603
#  gen: 177, stag: 64, mean: 116926.24, median: 124474.77, stdev: 80793.54, best: 304.3603
#  gen: 178, stag: 65, mean: 85051.40, median: 52752.58, stdev: 83387.09, best: 304.3603
#  gen: 179, stag: 66, mean: 53470.67, median: 29022.66, stdev: 59500.95, best: 304.3603
#  gen: 180, stag: 67, mean: 57662.59, median: 42572.83, stdev: 50001.62, best: 304.3603
#  gen: 181, stag: 68, mean: 72607.71, median: 72395.78, stdev: 57195.33, best: 304.3603
#  gen: 182, stag: 69, mean: 96920.19, median: 105492.05, stdev: 70894.69, best: 304.3603
#  gen: 183, stag: 70, mean: 86662.15, median: 73836.67, stdev: 77010.80, best: 304.3603
#  gen: 184, stag: 71, mean: 64010.94, median: 62142.27, stdev: 61117.25, best: 304.3603
#  gen: 185, stag: 72, mean: 70050.65, median: 68661.26, stdev: 62178.04, best: 304.3603
#  gen: 186, stag: 73, mean: 79045.14, median: 84106.45, stdev: 59590.32, best: 304.3603
#  gen: 187, stag: 74, mean: 81947.74, median: 78708.40, stdev: 61910.85, best: 304.3603
#  gen: 188, stag: 75, mean: 77341.39, median: 68215.96, stdev: 67847.71, best: 304.3603
#  gen: 189, stag: 76, mean: 82492.73, median: 78611.06, stdev: 70736.42, best: 304.3603
#  gen: 190, stag: 77, mean: 98739.56, median: 89592.31, stdev: 76430.21, best: 304.3603
#  gen: 191, stag: 78, mean: 86013.04, median: 86992.15, stdev: 63421.96, best: 304.3603
#  gen: 192, stag: 79, mean: 60578.87, median: 31781.04, stdev: 67010.50, best: 304.3603
#  gen: 193, stag: 80, mean: 43275.71, median: 26051.23, stdev: 47017.14, best: 304.3603
#  gen: 194, stag: 81, mean: 35836.11, median: 18815.62, stdev: 43570.68, best: 304.3603
#  gen: 195, stag: 82, mean: 60514.59, median: 42057.70, stdev: 63176.93, best: 304.3603
#  gen: 196, stag: 83, mean: 54321.52, median: 36211.22, stdev: 52046.23, best: 304.3603
#  gen: 197, stag: 84, mean: 52739.71, median: 39132.57, stdev: 54014.45, best: 304.3603
#  gen: 198, stag: 85, mean: 72267.28, median: 66305.28, stdev: 59708.06, best: 304.3603
#  gen: 199, stag: 86, mean: 33944.16, median: 13058.74, stdev: 42908.96, best: 304.3603
#  gen: 200, stag: 87, mean: 48492.15, median: 42606.36, stdev: 47188.05, best: 304.3603
#  gen: 201, stag: 88, mean: 51550.93, median: 39743.58, stdev: 53654.15, best: 304.3603
#  gen: 202, stag: 89, mean: 29787.93, median: 6995.66, stdev: 44875.97, best: 304.3603
#  gen: 203, stag: 90, mean: 45925.11, median: 19526.04, stdev: 60066.39, best: 304.3603
#  gen: 204, stag: 91, mean: 40690.87, median: 24964.91, stdev: 42304.00, best: 304.3603
#  gen: 205, stag: 92, mean: 34879.71, median: 11598.22, stdev: 46856.49, best: 304.3603
#  gen: 206, stag: 93, mean: 46216.19, median: 35325.00, stdev: 41752.96, best: 304.3603
#  gen: 207, stag: 94, mean: 45469.45, median: 63778.32, stdev: 37268.89, best: 304.3603
#  gen: 208, stag: 95, mean: 58525.53, median: 64334.98, stdev: 39612.33, best: 304.3603
#  gen: 209, stag: 96, mean: 58983.29, median: 69859.32, stdev: 37946.89, best: 304.3603
#  gen: 210, stag: 97, mean: 55819.73, median: 63378.64, stdev: 40913.78, best: 304.3603
#  gen: 211, stag: 0, mean: 50009.56, median: 59040.17, stdev: 43495.30, best: 161.0713
#  gen: 212, stag: 1, mean: 46762.71, median: 52420.93, stdev: 31584.49, best: 161.0713
#  gen: 213, stag: 2, mean: 56143.73, median: 60459.98, stdev: 46899.56, best: 161.0713
#  gen: 214, stag: 3, mean: 50999.19, median: 49788.42, stdev: 36948.38, best: 161.0713
#  gen: 215, stag: 4, mean: 60062.22, median: 63977.52, stdev: 54159.81, best: 161.0713
#  gen: 216, stag: 5, mean: 47426.02, median: 27840.54, stdev: 52243.59, best: 161.0713
#  gen: 217, stag: 6, mean: 49616.14, median: 34208.17, stdev: 46714.07, best: 161.0713
#  gen: 218, stag: 7, mean: 52696.02, median: 45538.09, stdev: 47227.32, best: 161.0713
#  gen: 219, stag: 8, mean: 47838.09, median: 33482.94, stdev: 49742.44, best: 161.0713
#  gen: 220, stag: 9, mean: 47628.06, median: 26241.21, stdev: 51555.88, best: 161.0713
#  gen: 221, stag: 0, mean: 47475.05, median: 33457.77, stdev: 50658.72, best: 117.4710
#  gen: 222, stag: 1, mean: 46619.21, median: 34248.68, stdev: 46605.84, best: 117.4710
#  gen: 223, stag: 2, mean: 53953.91, median: 43175.13, stdev: 51580.79, best: 117.4710
#  gen: 224, stag: 3, mean: 74369.54, median: 70137.86, stdev: 54609.11, best: 117.4710
#  gen: 225, stag: 4, mean: 99521.50, median: 111013.90, stdev: 68952.71, best: 117.4710
#  gen: 226, stag: 5, mean: 105553.44, median: 118776.24, stdev: 82270.86, best: 117.4710
#  gen: 227, stag: 6, mean: 112977.67, median: 118931.28, stdev: 76229.65, best: 117.4710
#  gen: 228, stag: 7, mean: 91517.28, median: 80468.75, stdev: 71467.26, best: 117.4710
#  gen: 229, stag: 8, mean: 72757.00, median: 64284.44, stdev: 65579.02, best: 117.4710
#  gen: 230, stag: 9, mean: 77249.90, median: 69003.45, stdev: 63135.72, best: 117.4710
#  gen: 231, stag: 10, mean: 73292.52, median: 62894.80, stdev: 70538.83, best: 117.4710
#  gen: 232, stag: 11, mean: 85271.13, median: 54456.66, stdev: 86801.71, best: 117.4710
#  gen: 233, stag: 12, mean: 86770.09, median: 87462.11, stdev: 70041.29, best: 117.4710
#  gen: 234, stag: 13, mean: 60773.97, median: 65637.56, stdev: 51992.44, best: 117.4710
#  gen: 235, stag: 14, mean: 57953.58, median: 60900.39, stdev: 52618.96, best: 117.4710
#  gen: 236, stag: 15, mean: 48915.73, median: 36664.69, stdev: 49152.74, best: 117.4710
#  gen: 237, stag: 16, mean: 62567.31, median: 52239.21, stdev: 58174.39, best: 117.4710
#  gen: 238, stag: 17, mean: 87106.03, median: 77673.01, stdev: 70394.00, best: 117.4710
#  gen: 239, stag: 18, mean: 85409.61, median: 71356.68, stdev: 68349.93, best: 117.4710
#  gen: 240, stag: 19, mean: 73628.90, median: 68839.74, stdev: 64097.98, best: 117.4710
#  gen: 241, stag: 20, mean: 63435.49, median: 47789.31, stdev: 61144.03, best: 117.4710
#  gen: 242, stag: 21, mean: 64588.49, median: 44834.72, stdev: 60511.03, best: 117.4710
#  gen: 243, stag: 22, mean: 79552.40, median: 71418.15, stdev: 66872.45, best: 117.4710
#  gen: 244, stag: 23, mean: 83936.04, median: 64081.18, stdev: 76158.80, best: 117.4710
#  gen: 245, stag: 24, mean: 65661.42, median: 43942.69, stdev: 69503.97, best: 117.4710
#  gen: 246, stag: 25, mean: 75846.10, median: 64578.25, stdev: 60250.57, best: 117.4710
#  gen: 247, stag: 26, mean: 77004.11, median: 81554.56, stdev: 61542.34, best: 117.4710
#  gen: 248, stag: 27, mean: 85972.51, median: 72695.43, stdev: 68339.42, best: 117.4710
#  gen: 249, stag: 28, mean: 70545.33, median: 64427.09, stdev: 65845.47, best: 117.4710
#  gen: 250, stag: 29, mean: 61408.80, median: 33486.77, stdev: 72001.10, best: 117.4710
#  gen: 251, stag: 30, mean: 49915.64, median: 8246.67, stdev: 70696.28, best: 117.4710
#  gen: 252, stag: 31, mean: 84135.65, median: 57148.74, stdev: 82244.13, best: 117.4710
#  gen: 253, stag: 32, mean: 80891.50, median: 64060.68, stdev: 77986.41, best: 117.4710
#  gen: 254, stag: 33, mean: 76328.15, median: 65802.42, stdev: 70725.04, best: 117.4710
#  gen: 255, stag: 34, mean: 98882.85, median: 83899.64, stdev: 85822.46, best: 117.4710
#  gen: 256, stag: 35, mean: 96628.32, median: 92519.80, stdev: 71048.22, best: 117.4710
#  gen: 257, stag: 36, mean: 95368.26, median: 90735.77, stdev: 70978.75, best: 117.4710
#  gen: 258, stag: 37, mean: 57519.92, median: 43767.15, stdev: 54395.92, best: 117.4710
#  gen: 259, stag: 38, mean: 64054.76, median: 44600.17, stdev: 59261.77, best: 117.4710
#  gen: 260, stag: 39, mean: 74236.52, median: 63101.01, stdev: 62928.71, best: 117.4710
#  gen: 261, stag: 40, mean: 59333.85, median: 58114.59, stdev: 57238.50, best: 117.4710
#  gen: 262, stag: 0, mean: 60027.32, median: 42414.26, stdev: 70217.85, best: 98.2084
#  gen: 263, stag: 1, mean: 48647.77, median: 15113.83, stdev: 73334.08, best: 98.2084
#  gen: 264, stag: 2, mean: 42297.15, median: 7075.71, stdev: 59262.66, best: 98.2084
#  gen: 265, stag: 3, mean: 60324.80, median: 31386.83, stdev: 69338.94, best: 98.2084
#  gen: 266, stag: 4, mean: 83455.44, median: 82663.51, stdev: 65302.04, best: 98.2084
#  gen: 267, stag: 5, mean: 97330.14, median: 93906.68, stdev: 71325.04, best: 98.2084
#  gen: 268, stag: 6, mean: 108777.01, median: 123384.03, stdev: 66518.35, best: 98.2084
#  gen: 269, stag: 7, mean: 102757.65, median: 95917.29, stdev: 68942.44, best: 98.2084
#  gen: 270, stag: 8, mean: 105574.10, median: 109695.00, stdev: 67228.73, best: 98.2084
#  gen: 271, stag: 9, mean: 67765.02, median: 33527.03, stdev: 71145.32, best: 98.2084
#  gen: 272, stag: 10, mean: 47476.34, median: 6322.78, stdev: 60152.67, best: 98.2084
#  gen: 273, stag: 0, mean: 65530.00, median: 53132.37, stdev: 58500.70, best: 92.0581
#  gen: 274, stag: 1, mean: 71461.89, median: 67220.04, stdev: 62478.25, best: 92.0581
#  gen: 275, stag: 2, mean: 60368.51, median: 47930.12, stdev: 49873.74, best: 92.0581
#  gen: 276, stag: 3, mean: 63065.16, median: 52595.64, stdev: 56144.63, best: 92.0581
#  gen: 277, stag: 4, mean: 68174.87, median: 49924.23, stdev: 58087.00, best: 92.0581
#  gen: 278, stag: 5, mean: 66705.56, median: 52150.99, stdev: 56248.28, best: 92.0581
#  gen: 279, stag: 6, mean: 71218.13, median: 66316.43, stdev: 57294.02, best: 92.0581
#  gen: 280, stag: 7, mean: 58425.64, median: 42569.31, stdev: 59121.95, best: 92.0581
#  gen: 281, stag: 8, mean: 53703.40, median: 43426.99, stdev: 50624.65, best: 92.0581
#  gen: 282, stag: 9, mean: 73735.31, median: 66051.98, stdev: 66699.41, best: 92.0581
#  gen: 283, stag: 10, mean: 82012.00, median: 78999.31, stdev: 64186.43, best: 92.0581
#  gen: 284, stag: 11, mean: 62629.35, median: 48613.63, stdev: 58747.36, best: 92.0581
#  gen: 285, stag: 12, mean: 41915.38, median: 23618.11, stdev: 52554.14, best: 92.0581
#  gen: 286, stag: 13, mean: 34219.68, median: 19830.41, stdev: 41447.21, best: 92.0581
#  gen: 287, stag: 14, mean: 32365.55, median: 9468.70, stdev: 48091.82, best: 92.0581
#  gen: 288, stag: 15, mean: 38157.72, median: 14599.12, stdev: 54355.88, best: 92.0581
#  gen: 289, stag: 16, mean: 41199.26, median: 36050.91, stdev: 42140.15, best: 92.0581
#  gen: 290, stag: 17, mean: 41000.71, median: 31597.27, stdev: 44412.79, best: 92.0581
#  gen: 291, stag: 18, mean: 59061.97, median: 57503.69, stdev: 47828.98, best: 92.0581
#  gen: 292, stag: 19, mean: 44077.02, median: 35170.44, stdev: 41085.52, best: 92.0581
#  gen: 293, stag: 20, mean: 43448.69, median: 22199.16, stdev: 51038.34, best: 92.0581
#  gen: 294, stag: 21, mean: 69170.54, median: 54363.92, stdev: 69220.85, best: 92.0581
#  gen: 295, stag: 22, mean: 68494.49, median: 56634.36, stdev: 67676.12, best: 92.0581
#  gen: 296, stag: 23, mean: 62894.11, median: 50049.61, stdev: 61256.46, best: 92.0581
#  gen: 297, stag: 24, mean: 43637.73, median: 28724.97, stdev: 49665.68, best: 92.0581
#  gen: 298, stag: 25, mean: 43635.22, median: 18346.33, stdev: 58769.29, best: 92.0581
#  gen: 299, stag: 26, mean: 42924.83, median: 30132.71, stdev: 48609.82, best: 92.0581
#  gen: 300, stag: 27, mean: 43848.23, median: 13615.32, stdev: 58809.49, best: 92.0581
#  gen: 301, stag: 28, mean: 54250.64, median: 30393.43, stdev: 62297.31, best: 92.0581

#  Total generations: 301
#  Total CPU time: 8.355765586999999 s
#  Best score: 92.05805979312564
#  Real chromosome of best solution: [0.04258633 0.6209769  0.17158706 0.04221758 0.3004926  0.4072837
#  0.68210946]
#  Binary chromosome of best solution: []
