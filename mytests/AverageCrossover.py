from evolvekit import *
from evolvekit.operators.Ga.real.mutation.UniformMutation import UniformMutation
from evolvekit.operators.Ga.real.crossover.AverageCrossover import AverageCrossover


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
crossover_real = AverageCrossover()
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

# (venv) (base) rafal@rafal-Lenovo-ideapad-720-15IKB:~/Documents/studia magisterskie/genesys/EvolveKit$  cd /home/rafal/Documents/studia\ magisterskie/genesys/EvolveKit ; /usr/bin/env /home/rafal/Documents/studia\ magisterskie/genesys/EvolveKit/venv/bin/python /home/rafal/.vscode/extensions/ms-python.debugpy-2025.14.1-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 59101 -- /home/rafal/Documents/studia\ magisterskie/genesys/EvolveKit/mytests/AverageCrossover.py

#  >>> EvolveKit Genetic Algorithm Demo <<<

#  gen: 1, stag: 0, mean: 95094.78, median: 85764.14, stdev: 63690.07, best: 6656.6039
#  gen: 2, stag: 0, mean: 36176.57, median: 28628.24, stdev: 31429.84, best: 906.3542
#  gen: 3, stag: 0, mean: 20280.98, median: 12069.95, stdev: 17352.10, best: 797.4634
#  gen: 4, stag: 1, mean: 16984.00, median: 9352.71, stdev: 18720.14, best: 797.4634
#  gen: 5, stag: 2, mean: 27669.85, median: 22519.19, stdev: 23901.56, best: 797.4634
#  gen: 6, stag: 3, mean: 20158.97, median: 22157.26, stdev: 10707.68, best: 797.4634
#  gen: 7, stag: 4, mean: 20164.58, median: 23189.65, stdev: 9741.14, best: 797.4634
#  gen: 8, stag: 5, mean: 21902.42, median: 25430.40, stdev: 10805.99, best: 797.4634
#  gen: 9, stag: 6, mean: 22452.54, median: 26795.94, stdev: 10402.32, best: 797.4634
#  gen: 10, stag: 7, mean: 20124.07, median: 27428.82, stdev: 11871.06, best: 797.4634
#  gen: 11, stag: 8, mean: 20716.32, median: 28448.20, stdev: 12074.56, best: 797.4634
#  gen: 12, stag: 9, mean: 19533.96, median: 28584.40, stdev: 12125.00, best: 797.4634
#  gen: 13, stag: 10, mean: 20270.15, median: 28577.04, stdev: 11551.50, best: 797.4634
#  gen: 14, stag: 11, mean: 23499.48, median: 29122.84, stdev: 10862.47, best: 797.4634
#  gen: 15, stag: 12, mean: 20434.29, median: 29311.40, stdev: 12478.74, best: 797.4634
#  gen: 16, stag: 13, mean: 19661.83, median: 29311.40, stdev: 12297.96, best: 797.4634
#  gen: 17, stag: 14, mean: 21239.50, median: 29374.17, stdev: 11512.00, best: 797.4634
#  gen: 18, stag: 15, mean: 22656.47, median: 29400.04, stdev: 10842.80, best: 797.4634
#  gen: 19, stag: 16, mean: 23104.97, median: 29389.38, stdev: 10850.72, best: 797.4634
#  gen: 20, stag: 17, mean: 23710.47, median: 29424.70, stdev: 11033.75, best: 797.4634
#  gen: 21, stag: 18, mean: 23959.83, median: 29433.09, stdev: 11160.14, best: 797.4634
#  gen: 22, stag: 19, mean: 21868.02, median: 29436.23, stdev: 15052.28, best: 797.4634
#  gen: 23, stag: 20, mean: 20334.36, median: 29436.50, stdev: 11135.35, best: 797.4634
#  gen: 24, stag: 21, mean: 21553.96, median: 29438.44, stdev: 11301.53, best: 797.4634
#  gen: 25, stag: 22, mean: 23874.67, median: 29439.26, stdev: 11109.60, best: 797.4634
#  gen: 26, stag: 23, mean: 23884.13, median: 29439.37, stdev: 11113.93, best: 797.4634
#  gen: 27, stag: 24, mean: 23884.25, median: 29439.56, stdev: 11113.99, best: 797.4634
#  gen: 28, stag: 25, mean: 24200.11, median: 29439.69, stdev: 11706.55, best: 797.4634
#  gen: 29, stag: 26, mean: 24568.14, median: 29439.72, stdev: 11920.66, best: 797.4634
#  gen: 30, stag: 27, mean: 24013.68, median: 29439.81, stdev: 14793.89, best: 797.4634
#  gen: 31, stag: 28, mean: 22177.61, median: 29439.82, stdev: 12204.83, best: 797.4634
#  gen: 32, stag: 29, mean: 20720.91, median: 22584.61, stdev: 11519.74, best: 797.4634
#  gen: 33, stag: 30, mean: 24386.15, median: 29742.09, stdev: 11884.36, best: 797.4634
#  gen: 34, stag: 31, mean: 25476.01, median: 29952.82, stdev: 12203.86, best: 797.4634
#  gen: 35, stag: 32, mean: 25183.29, median: 30330.60, stdev: 11898.67, best: 797.4634
#  gen: 36, stag: 33, mean: 25419.82, median: 30902.66, stdev: 11943.07, best: 797.4634
#  gen: 37, stag: 34, mean: 25251.42, median: 31281.10, stdev: 12016.15, best: 797.4634
#  gen: 38, stag: 35, mean: 25864.85, median: 31886.72, stdev: 12126.32, best: 797.4634
#  gen: 39, stag: 36, mean: 26148.91, median: 32072.41, stdev: 12274.75, best: 797.4634
#  gen: 40, stag: 37, mean: 26568.48, median: 32072.41, stdev: 12512.84, best: 797.4634
#  gen: 41, stag: 38, mean: 26725.04, median: 32342.76, stdev: 12574.41, best: 797.4634
#  gen: 42, stag: 39, mean: 21222.16, median: 32307.46, stdev: 14419.87, best: 797.4634
#  gen: 43, stag: 40, mean: 20775.38, median: 32642.76, stdev: 13464.15, best: 797.4634
#  gen: 44, stag: 41, mean: 25501.63, median: 33242.85, stdev: 12857.52, best: 797.4634
#  gen: 45, stag: 42, mean: 27033.69, median: 33347.33, stdev: 12730.46, best: 797.4634
#  gen: 46, stag: 43, mean: 27041.76, median: 33361.99, stdev: 12734.42, best: 797.4634
#  gen: 47, stag: 44, mean: 27104.88, median: 33461.97, stdev: 12765.74, best: 797.4634
#  gen: 48, stag: 45, mean: 25874.64, median: 33479.53, stdev: 15727.72, best: 797.4634
#  gen: 49, stag: 46, mean: 27145.48, median: 33526.07, stdev: 12788.08, best: 797.4634
#  gen: 50, stag: 47, mean: 27370.75, median: 33537.09, stdev: 13051.92, best: 797.4634
#  gen: 51, stag: 48, mean: 27158.92, median: 33546.41, stdev: 12794.66, best: 797.4634
#  gen: 52, stag: 49, mean: 23804.36, median: 33541.06, stdev: 14278.72, best: 797.4634
#  gen: 53, stag: 50, mean: 24440.04, median: 33544.20, stdev: 13235.65, best: 797.4634
#  gen: 54, stag: 51, mean: 22141.00, median: 33548.23, stdev: 14291.96, best: 797.4634
#  gen: 55, stag: 52, mean: 18648.67, median: 14847.50, stdev: 14100.76, best: 797.4634
#  gen: 56, stag: 53, mean: 24271.78, median: 33552.95, stdev: 12688.10, best: 797.4634
#  gen: 57, stag: 54, mean: 24683.80, median: 33554.82, stdev: 18324.73, best: 797.4634
#  gen: 58, stag: 55, mean: 25297.08, median: 33555.84, stdev: 13033.33, best: 797.4634
#  gen: 59, stag: 56, mean: 22673.28, median: 25054.33, stdev: 12371.96, best: 797.4634
#  gen: 60, stag: 57, mean: 19962.27, median: 25054.10, stdev: 12048.37, best: 797.4634
#  gen: 61, stag: 58, mean: 23245.31, median: 29161.93, stdev: 11043.96, best: 797.4634
#  gen: 62, stag: 59, mean: 24695.86, median: 29197.29, stdev: 13822.99, best: 797.4634
#  gen: 63, stag: 60, mean: 22394.58, median: 29197.36, stdev: 12116.66, best: 797.4634
#  gen: 64, stag: 61, mean: 24187.18, median: 29473.08, stdev: 11367.72, best: 797.4634
#  gen: 65, stag: 62, mean: 24101.27, median: 29611.69, stdev: 11308.34, best: 797.4634
#  gen: 66, stag: 63, mean: 24138.43, median: 29750.82, stdev: 11325.24, best: 797.4634
#  gen: 67, stag: 64, mean: 25519.21, median: 29872.99, stdev: 17486.83, best: 797.4634
#  gen: 68, stag: 65, mean: 21093.02, median: 29942.97, stdev: 12520.53, best: 797.4634
#  gen: 69, stag: 66, mean: 23459.69, median: 29960.49, stdev: 11445.82, best: 797.4634
#  gen: 70, stag: 67, mean: 20351.55, median: 29973.63, stdev: 12502.81, best: 797.4634
#  gen: 71, stag: 68, mean: 16419.12, median: 16324.05, stdev: 11584.84, best: 797.4634
#  gen: 72, stag: 69, mean: 12755.29, median: 11647.04, stdev: 8830.29, best: 797.4634
#  gen: 73, stag: 70, mean: 19404.04, median: 24100.73, stdev: 10264.20, best: 797.4634
#  gen: 74, stag: 71, mean: 19084.32, median: 24100.27, stdev: 11055.46, best: 797.4634
#  gen: 75, stag: 72, mean: 20645.80, median: 27420.85, stdev: 11478.26, best: 797.4634
#  gen: 76, stag: 73, mean: 21640.54, median: 27925.44, stdev: 10820.89, best: 797.4634
#  gen: 77, stag: 74, mean: 22897.26, median: 28431.82, stdev: 10890.02, best: 797.4634
#  gen: 78, stag: 75, mean: 20238.11, median: 28560.17, stdev: 12267.54, best: 797.4634
#  gen: 79, stag: 76, mean: 23443.07, median: 28819.06, stdev: 11138.19, best: 797.4634
#  gen: 80, stag: 77, mean: 20889.37, median: 28895.31, stdev: 12036.29, best: 797.4634
#  gen: 81, stag: 78, mean: 23625.76, median: 29110.80, stdev: 11069.12, best: 797.4634
#  gen: 82, stag: 79, mean: 22081.35, median: 29166.97, stdev: 14816.84, best: 797.4634
#  gen: 83, stag: 80, mean: 15403.66, median: 7979.69, stdev: 12483.62, best: 797.4634
#  gen: 84, stag: 81, mean: 18835.11, median: 29193.62, stdev: 11783.22, best: 797.4634
#  gen: 85, stag: 82, mean: 21534.11, median: 29224.67, stdev: 12117.29, best: 797.4634
#  gen: 86, stag: 83, mean: 22197.85, median: 28499.87, stdev: 11650.93, best: 797.4634
#  gen: 87, stag: 84, mean: 17046.14, median: 21839.11, stdev: 11806.02, best: 797.4634
#  gen: 88, stag: 85, mean: 21154.27, median: 29276.59, stdev: 11079.11, best: 797.4634
#  gen: 89, stag: 86, mean: 24293.49, median: 29348.18, stdev: 11848.77, best: 797.4634
#  gen: 90, stag: 87, mean: 20997.27, median: 29374.09, stdev: 13122.71, best: 797.4634
#  gen: 91, stag: 88, mean: 24643.50, median: 29411.51, stdev: 11747.86, best: 797.4634
#  gen: 92, stag: 89, mean: 24995.07, median: 29575.05, stdev: 11882.69, best: 797.4634
#  gen: 93, stag: 90, mean: 26612.46, median: 32172.06, stdev: 12656.37, best: 797.4634
#  gen: 94, stag: 91, mean: 28115.08, median: 33980.06, stdev: 14309.36, best: 797.4634
#  gen: 95, stag: 92, mean: 27824.40, median: 34483.27, stdev: 13436.68, best: 797.4634
#  gen: 96, stag: 93, mean: 25289.01, median: 34230.93, stdev: 14095.18, best: 797.4634
#  gen: 97, stag: 94, mean: 25587.30, median: 34490.68, stdev: 13505.74, best: 797.4634
#  gen: 98, stag: 95, mean: 23962.69, median: 34655.37, stdev: 14611.88, best: 797.4634
#  gen: 99, stag: 96, mean: 19161.76, median: 18702.73, stdev: 14534.81, best: 797.4634
#  gen: 100, stag: 97, mean: 19703.64, median: 18768.63, stdev: 12737.63, best: 797.4634
#  gen: 101, stag: 98, mean: 21134.06, median: 18768.63, stdev: 12649.42, best: 797.4634
#  gen: 102, stag: 99, mean: 25587.88, median: 34808.41, stdev: 12939.29, best: 797.4634
#  gen: 103, stag: 100, mean: 26561.18, median: 34806.32, stdev: 14216.73, best: 797.4634
#  gen: 104, stag: 101, mean: 27288.31, median: 33594.43, stdev: 13008.34, best: 797.4634
#  gen: 105, stag: 102, mean: 27489.88, median: 33586.76, stdev: 13065.49, best: 797.4634
#  gen: 106, stag: 103, mean: 27473.42, median: 34202.60, stdev: 13204.46, best: 797.4634
#  gen: 107, stag: 104, mean: 18796.03, median: 9212.03, stdev: 14893.63, best: 797.4634
#  gen: 108, stag: 105, mean: 14932.19, median: 16398.18, stdev: 10611.85, best: 797.4634
#  gen: 109, stag: 106, mean: 16332.84, median: 18186.73, stdev: 11154.00, best: 797.4634
#  gen: 110, stag: 107, mean: 12008.20, median: 8960.05, stdev: 10251.35, best: 797.4634
#  gen: 111, stag: 108, mean: 11653.86, median: 11320.89, stdev: 6940.09, best: 797.4634
#  gen: 112, stag: 109, mean: 13601.96, median: 15358.93, stdev: 6615.91, best: 797.4634
#  gen: 113, stag: 110, mean: 14196.57, median: 16850.73, stdev: 6576.08, best: 797.4634
#  gen: 114, stag: 111, mean: 13631.03, median: 17755.68, stdev: 7505.56, best: 797.4634
#  gen: 115, stag: 112, mean: 13986.83, median: 17856.29, stdev: 6920.12, best: 797.4634
#  gen: 116, stag: 113, mean: 15652.56, median: 18235.35, stdev: 9531.94, best: 797.4634
#  gen: 117, stag: 114, mean: 13328.76, median: 18239.18, stdev: 10295.27, best: 797.4634
#  gen: 118, stag: 115, mean: 13108.26, median: 17140.20, stdev: 9832.01, best: 797.4634
#  gen: 119, stag: 116, mean: 14253.58, median: 17419.40, stdev: 6720.81, best: 797.4634
#  gen: 120, stag: 117, mean: 14687.50, median: 17357.03, stdev: 8191.35, best: 797.4634
#  gen: 121, stag: 118, mean: 14268.00, median: 17317.79, stdev: 6641.39, best: 797.4634
#  gen: 122, stag: 119, mean: 14388.54, median: 17271.04, stdev: 6702.39, best: 797.4634
#  gen: 123, stag: 120, mean: 12010.41, median: 14205.95, stdev: 11433.77, best: 797.4634
#  gen: 124, stag: 121, mean: 18520.72, median: 15601.81, stdev: 19565.84, best: 797.4634
#  gen: 125, stag: 122, mean: 20542.82, median: 19788.42, stdev: 13479.20, best: 797.4634
#  gen: 126, stag: 123, mean: 17866.67, median: 19806.26, stdev: 9134.20, best: 797.4634
#  gen: 127, stag: 124, mean: 18226.85, median: 21746.57, stdev: 8703.55, best: 797.4634
#  gen: 128, stag: 125, mean: 17560.48, median: 20640.75, stdev: 8264.67, best: 797.4634
#  gen: 129, stag: 126, mean: 17381.52, median: 20169.77, stdev: 8429.73, best: 797.4634
#  gen: 130, stag: 127, mean: 18354.54, median: 21223.05, stdev: 10849.23, best: 797.4634
#  gen: 131, stag: 128, mean: 10988.83, median: 5519.43, stdev: 8959.82, best: 797.4634
#  gen: 132, stag: 129, mean: 11617.84, median: 11224.47, stdev: 7733.91, best: 797.4634
#  gen: 133, stag: 130, mean: 11127.95, median: 11224.47, stdev: 6229.06, best: 797.4634
#  gen: 134, stag: 131, mean: 13756.96, median: 15847.35, stdev: 6399.59, best: 797.4634
#  gen: 135, stag: 132, mean: 13911.95, median: 16517.87, stdev: 10473.80, best: 797.4634
#  gen: 136, stag: 133, mean: 12399.58, median: 17036.01, stdev: 7475.60, best: 797.4634
#  gen: 137, stag: 134, mean: 9965.49, median: 9690.92, stdev: 7513.59, best: 797.4634
#  gen: 138, stag: 135, mean: 12501.60, median: 10241.28, stdev: 7107.47, best: 797.4634
#  gen: 139, stag: 136, mean: 12604.93, median: 14076.13, stdev: 10136.66, best: 797.4634
#  gen: 140, stag: 137, mean: 8865.66, median: 7844.40, stdev: 6984.68, best: 797.4634
#  gen: 141, stag: 138, mean: 12418.55, median: 14635.62, stdev: 6334.13, best: 797.4634
#  gen: 142, stag: 139, mean: 11827.72, median: 16449.66, stdev: 7381.88, best: 797.4634
#  gen: 143, stag: 140, mean: 12384.32, median: 16872.67, stdev: 7882.10, best: 797.4634
#  gen: 144, stag: 141, mean: 11030.47, median: 9093.55, stdev: 7743.87, best: 797.4634
#  gen: 145, stag: 142, mean: 12636.00, median: 16785.37, stdev: 7210.48, best: 797.4634
#  gen: 146, stag: 143, mean: 14345.29, median: 17635.57, stdev: 6728.31, best: 797.4634
#  gen: 147, stag: 144, mean: 14351.69, median: 17636.13, stdev: 6731.79, best: 797.4634
#  gen: 148, stag: 145, mean: 10352.98, median: 17652.76, stdev: 7765.23, best: 797.4634
#  gen: 149, stag: 146, mean: 9364.61, median: 6403.20, stdev: 7818.38, best: 797.4634
#  gen: 150, stag: 147, mean: 12421.47, median: 17706.08, stdev: 6601.65, best: 797.4634
#  gen: 151, stag: 148, mean: 11027.74, median: 14969.90, stdev: 7228.69, best: 797.4634
#  gen: 152, stag: 149, mean: 11231.61, median: 14969.90, stdev: 6942.66, best: 797.4634
#  gen: 153, stag: 150, mean: 11163.56, median: 16306.23, stdev: 6853.62, best: 797.4634
#  gen: 154, stag: 151, mean: 9143.29, median: 8181.40, stdev: 6249.73, best: 797.4634
#  gen: 155, stag: 152, mean: 7305.65, median: 8375.68, stdev: 5162.76, best: 797.4634
#  gen: 156, stag: 153, mean: 8422.49, median: 8375.68, stdev: 7007.30, best: 797.4634
#  gen: 157, stag: 154, mean: 10821.33, median: 12466.13, stdev: 6507.14, best: 797.4634
#  gen: 158, stag: 155, mean: 11369.69, median: 13153.13, stdev: 5465.52, best: 797.4634
#  gen: 159, stag: 156, mean: 7065.42, median: 3827.40, stdev: 6632.07, best: 797.4634
#  gen: 160, stag: 157, mean: 7846.42, median: 7809.28, stdev: 5376.11, best: 797.4634
#  gen: 161, stag: 158, mean: 5168.50, median: 3210.65, stdev: 4390.61, best: 797.4634
#  gen: 162, stag: 159, mean: 5883.33, median: 5454.13, stdev: 3817.76, best: 797.4634
#  gen: 163, stag: 160, mean: 6163.48, median: 5155.26, stdev: 3931.92, best: 797.4634
#  gen: 164, stag: 161, mean: 5856.66, median: 6869.00, stdev: 3646.54, best: 797.4634
#  gen: 165, stag: 162, mean: 6235.84, median: 7302.87, stdev: 3336.24, best: 797.4634
#  gen: 166, stag: 163, mean: 6203.39, median: 8353.63, stdev: 3942.15, best: 797.4634
#  gen: 167, stag: 164, mean: 7117.24, median: 10039.82, stdev: 3865.53, best: 797.4634
#  gen: 168, stag: 165, mean: 6574.22, median: 10090.00, stdev: 4083.24, best: 797.4634
#  gen: 169, stag: 166, mean: 6300.73, median: 8030.72, stdev: 3812.02, best: 797.4634
#  gen: 170, stag: 167, mean: 6745.67, median: 9027.84, stdev: 3670.62, best: 797.4634
#  gen: 171, stag: 168, mean: 7665.74, median: 9621.71, stdev: 3580.55, best: 797.4634
#  gen: 172, stag: 169, mean: 7241.71, median: 8304.76, stdev: 3322.98, best: 797.4634
#  gen: 173, stag: 170, mean: 7796.92, median: 8780.05, stdev: 4255.59, best: 797.4634
#  gen: 174, stag: 171, mean: 7821.65, median: 9618.69, stdev: 3505.93, best: 797.4634
#  gen: 175, stag: 172, mean: 7836.56, median: 9481.80, stdev: 3508.97, best: 797.4634
#  gen: 176, stag: 173, mean: 5740.27, median: 2959.52, stdev: 4373.51, best: 797.4634
#  gen: 177, stag: 174, mean: 5938.81, median: 5407.10, stdev: 3486.84, best: 797.4634
#  gen: 178, stag: 175, mean: 6321.13, median: 7268.90, stdev: 3620.03, best: 797.4634
#  gen: 179, stag: 176, mean: 6641.69, median: 7294.00, stdev: 3192.35, best: 797.4634
#  gen: 180, stag: 177, mean: 4979.37, median: 6251.10, stdev: 3130.85, best: 797.4634
#  gen: 181, stag: 178, mean: 4188.68, median: 4377.05, stdev: 2635.01, best: 797.4634
#  gen: 182, stag: 179, mean: 4536.95, median: 5358.17, stdev: 2656.31, best: 797.4634
#  gen: 183, stag: 180, mean: 5006.45, median: 6074.74, stdev: 2501.89, best: 797.4634
#  gen: 184, stag: 181, mean: 5043.96, median: 6160.09, stdev: 2268.67, best: 797.4634
#  gen: 185, stag: 182, mean: 5344.12, median: 6322.48, stdev: 2611.27, best: 797.4634
#  gen: 186, stag: 183, mean: 5536.08, median: 6289.23, stdev: 3608.97, best: 797.4634
#  gen: 187, stag: 184, mean: 5365.76, median: 6461.86, stdev: 2266.19, best: 797.4634
#  gen: 188, stag: 185, mean: 5457.25, median: 6552.13, stdev: 2311.49, best: 797.4634
#  gen: 189, stag: 186, mean: 5538.91, median: 6660.78, stdev: 2358.59, best: 797.4634
#  gen: 190, stag: 187, mean: 5536.24, median: 6700.98, stdev: 2347.48, best: 797.4634
#  gen: 191, stag: 188, mean: 5069.05, median: 6699.83, stdev: 2574.99, best: 797.4634
#  gen: 192, stag: 189, mean: 3907.45, median: 2845.59, stdev: 2609.75, best: 797.4634
#  gen: 193, stag: 190, mean: 3868.15, median: 3302.10, stdev: 2145.98, best: 797.4634
#  gen: 194, stag: 191, mean: 3913.01, median: 4021.86, stdev: 1778.29, best: 797.4634
#  gen: 195, stag: 192, mean: 4440.86, median: 5196.33, stdev: 1943.03, best: 797.4634
#  gen: 196, stag: 193, mean: 5067.71, median: 6071.97, stdev: 2157.69, best: 797.4634
#  gen: 197, stag: 194, mean: 5128.28, median: 6076.01, stdev: 2172.09, best: 797.4634
#  gen: 198, stag: 195, mean: 5431.57, median: 6080.56, stdev: 8815.18, best: 797.4634
#  gen: 199, stag: 196, mean: 5492.72, median: 6249.60, stdev: 7164.28, best: 797.4634
#  gen: 200, stag: 197, mean: 4619.27, median: 6344.93, stdev: 2476.14, best: 797.4634
#  gen: 201, stag: 198, mean: 5302.27, median: 6410.42, stdev: 2247.43, best: 797.4634
#  gen: 202, stag: 199, mean: 4609.89, median: 6394.03, stdev: 2467.73, best: 797.4634
#  gen: 203, stag: 200, mean: 4437.26, median: 3972.26, stdev: 2296.37, best: 797.4634
#  gen: 204, stag: 201, mean: 4440.16, median: 5056.62, stdev: 2065.42, best: 797.4634
#  gen: 205, stag: 202, mean: 5536.43, median: 5707.65, stdev: 11054.58, best: 797.4634
#  gen: 206, stag: 203, mean: 4850.37, median: 5712.13, stdev: 2061.18, best: 797.4634
#  gen: 207, stag: 204, mean: 5012.67, median: 5884.96, stdev: 2124.33, best: 797.4634
#  gen: 208, stag: 205, mean: 5454.89, median: 6150.55, stdev: 7930.84, best: 797.4634
#  gen: 209, stag: 206, mean: 4832.75, median: 6149.01, stdev: 2162.64, best: 797.4634
#  gen: 210, stag: 207, mean: 4928.53, median: 6150.33, stdev: 2109.14, best: 797.4634
#  gen: 211, stag: 208, mean: 4597.41, median: 5493.34, stdev: 4201.98, best: 797.4634
#  gen: 212, stag: 209, mean: 5112.90, median: 6194.55, stdev: 2175.01, best: 797.4634
#  gen: 213, stag: 210, mean: 4917.78, median: 6194.58, stdev: 3343.45, best: 797.4634
#  gen: 214, stag: 211, mean: 4683.49, median: 6202.91, stdev: 2632.33, best: 797.4634
#  gen: 215, stag: 212, mean: 3951.97, median: 4275.58, stdev: 2244.61, best: 797.4634
#  gen: 216, stag: 213, mean: 4815.06, median: 6202.96, stdev: 3949.92, best: 797.4634
#  gen: 217, stag: 214, mean: 4760.86, median: 6208.26, stdev: 2205.28, best: 797.4634
#  gen: 218, stag: 215, mean: 5171.05, median: 6208.00, stdev: 2184.44, best: 797.4634
#  gen: 219, stag: 216, mean: 5189.89, median: 6252.74, stdev: 2193.45, best: 797.4634
#  gen: 220, stag: 217, mean: 4486.60, median: 6264.37, stdev: 3248.03, best: 797.4634
#  gen: 221, stag: 218, mean: 4686.28, median: 6288.30, stdev: 2209.56, best: 797.4634
#  gen: 222, stag: 219, mean: 5219.81, median: 6313.89, stdev: 2208.09, best: 797.4634
#  gen: 223, stag: 220, mean: 5235.44, median: 6329.94, stdev: 2217.56, best: 797.4634
#  gen: 224, stag: 221, mean: 5257.77, median: 6339.03, stdev: 2239.88, best: 797.4634
#  gen: 225, stag: 222, mean: 5225.22, median: 6343.24, stdev: 2213.88, best: 797.4634
#  gen: 226, stag: 223, mean: 5237.68, median: 6345.73, stdev: 2216.96, best: 797.4634
#  gen: 227, stag: 224, mean: 5238.61, median: 6346.59, stdev: 2217.43, best: 797.4634
#  gen: 228, stag: 225, mean: 4121.72, median: 6346.69, stdev: 2535.88, best: 797.4634
#  gen: 229, stag: 226, mean: 4332.03, median: 6347.27, stdev: 2491.88, best: 797.4634
#  gen: 230, stag: 227, mean: 5237.74, median: 6347.69, stdev: 2220.14, best: 797.4634
#  gen: 231, stag: 228, mean: 5237.77, median: 6347.78, stdev: 2220.15, best: 797.4634
#  gen: 232, stag: 229, mean: 5367.51, median: 6347.91, stdev: 2646.96, best: 797.4634
#  gen: 233, stag: 230, mean: 5237.88, median: 6347.97, stdev: 2220.21, best: 797.4634
#  gen: 234, stag: 231, mean: 5226.84, median: 6347.97, stdev: 2217.41, best: 797.4634
#  gen: 235, stag: 232, mean: 5254.57, median: 6347.99, stdev: 2234.70, best: 797.4634
#  gen: 236, stag: 233, mean: 3943.02, median: 1872.01, stdev: 3313.86, best: 797.4634
#  gen: 237, stag: 234, mean: 4079.12, median: 6348.00, stdev: 2411.69, best: 797.4634
#  gen: 238, stag: 235, mean: 4425.55, median: 6348.00, stdev: 2340.77, best: 797.4634
#  gen: 239, stag: 236, mean: 5060.51, median: 6348.01, stdev: 2172.88, best: 797.4634
#  gen: 240, stag: 237, mean: 5237.83, median: 6348.02, stdev: 2220.18, best: 797.4634
#  gen: 241, stag: 238, mean: 5313.72, median: 6348.02, stdev: 2380.47, best: 797.4634
#  gen: 242, stag: 239, mean: 4579.03, median: 6348.02, stdev: 2482.03, best: 797.4634
#  gen: 243, stag: 240, mean: 5237.91, median: 6348.02, stdev: 2220.22, best: 797.4634
#  gen: 244, stag: 241, mean: 4389.25, median: 6348.02, stdev: 2522.33, best: 797.4634
#  gen: 245, stag: 242, mean: 4523.91, median: 6348.02, stdev: 4418.49, best: 797.4634
#  gen: 246, stag: 243, mean: 4201.25, median: 4357.95, stdev: 2333.11, best: 797.4634
#  gen: 247, stag: 244, mean: 4497.97, median: 6348.02, stdev: 2786.89, best: 797.4634
#  gen: 248, stag: 245, mean: 5074.00, median: 6348.02, stdev: 3597.38, best: 797.4634
#  gen: 249, stag: 246, mean: 4170.05, median: 4357.95, stdev: 2342.54, best: 797.4634
#  gen: 250, stag: 247, mean: 3841.62, median: 3153.83, stdev: 2173.80, best: 797.4634
#  gen: 251, stag: 248, mean: 3435.26, median: 3153.83, stdev: 2022.92, best: 797.4634
#  gen: 252, stag: 249, mean: 3274.26, median: 2738.57, stdev: 6850.70, best: 797.4634
#  gen: 253, stag: 250, mean: 2991.54, median: 3401.09, stdev: 1314.54, best: 797.4634
#  gen: 254, stag: 251, mean: 2780.32, median: 3213.77, stdev: 1102.56, best: 797.4634
#  gen: 255, stag: 252, mean: 2749.37, median: 3273.70, stdev: 1033.48, best: 797.4634
#  gen: 256, stag: 253, mean: 2835.87, median: 3368.51, stdev: 1038.97, best: 797.4634
#  gen: 257, stag: 254, mean: 2627.69, median: 3401.09, stdev: 1148.43, best: 797.4634
#  gen: 258, stag: 255, mean: 2930.68, median: 3434.18, stdev: 1087.21, best: 797.4634
#  gen: 259, stag: 256, mean: 2901.74, median: 3455.13, stdev: 1056.74, best: 797.4634
#  gen: 260, stag: 257, mean: 2958.71, median: 3476.27, stdev: 1091.24, best: 797.4634
#  gen: 261, stag: 258, mean: 2942.65, median: 3483.72, stdev: 1073.89, best: 797.4634
#  gen: 262, stag: 259, mean: 2946.73, median: 3486.92, stdev: 1075.17, best: 797.4634
#  gen: 263, stag: 260, mean: 2891.16, median: 3488.26, stdev: 1052.41, best: 797.4634
#  gen: 264, stag: 261, mean: 2939.89, median: 3491.13, stdev: 1072.16, best: 797.4634
#  gen: 265, stag: 262, mean: 2953.63, median: 3492.40, stdev: 1078.09, best: 797.4634
#  gen: 266, stag: 263, mean: 3026.02, median: 3492.33, stdev: 1326.98, best: 797.4634
#  gen: 267, stag: 264, mean: 2722.15, median: 3492.75, stdev: 1167.14, best: 797.4634
#  gen: 268, stag: 265, mean: 2836.26, median: 3492.92, stdev: 1078.58, best: 797.4634
#  gen: 269, stag: 266, mean: 3799.70, median: 3492.93, stdev: 7765.61, best: 797.4634
#  gen: 270, stag: 267, mean: 3657.12, median: 3492.92, stdev: 7650.56, best: 797.4634
#  gen: 271, stag: 268, mean: 5451.50, median: 3492.96, stdev: 8499.46, best: 797.4634
#  gen: 272, stag: 269, mean: 7244.29, median: 5066.07, stdev: 7788.07, best: 797.4634
#  gen: 273, stag: 270, mean: 8770.64, median: 11671.83, stdev: 4987.18, best: 797.4634
#  gen: 274, stag: 271, mean: 10793.75, median: 12513.84, stdev: 5889.67, best: 797.4634
#  gen: 275, stag: 272, mean: 8383.28, median: 13406.81, stdev: 7403.63, best: 797.4634
#  gen: 276, stag: 273, mean: 9236.86, median: 13406.81, stdev: 5824.77, best: 797.4634
#  gen: 277, stag: 274, mean: 8965.76, median: 13406.82, stdev: 5581.39, best: 797.4634
#  gen: 278, stag: 275, mean: 10884.95, median: 13406.82, stdev: 5043.74, best: 797.4634
#  gen: 279, stag: 276, mean: 8499.64, median: 13406.82, stdev: 6014.33, best: 797.4634
#  gen: 280, stag: 277, mean: 6349.40, median: 4357.69, stdev: 5456.38, best: 797.4634
#  gen: 281, stag: 278, mean: 10084.52, median: 13406.82, stdev: 5091.18, best: 797.4634
#  gen: 282, stag: 279, mean: 10410.20, median: 13406.82, stdev: 4931.74, best: 797.4634
#  gen: 283, stag: 280, mean: 9744.29, median: 11732.92, stdev: 4593.12, best: 797.4634
#  gen: 284, stag: 281, mean: 9843.01, median: 11732.92, stdev: 4599.22, best: 797.4634
#  gen: 285, stag: 282, mean: 7101.19, median: 11343.58, stdev: 5562.84, best: 797.4634
#  gen: 286, stag: 283, mean: 4261.72, median: 1476.13, stdev: 4927.93, best: 797.4634
#  gen: 287, stag: 0, mean: 5355.29, median: 2394.79, stdev: 4953.46, best: 707.4524
#  gen: 288, stag: 1, mean: 5469.41, median: 3669.52, stdev: 5278.43, best: 707.4524
#  gen: 289, stag: 2, mean: 5864.64, median: 5564.67, stdev: 3944.68, best: 707.4524
#  gen: 290, stag: 3, mean: 6255.80, median: 6881.33, stdev: 3372.61, best: 707.4524
#  gen: 291, stag: 4, mean: 6254.43, median: 6950.79, stdev: 2946.25, best: 707.4524
#  gen: 292, stag: 5, mean: 6793.40, median: 7685.90, stdev: 4204.01, best: 707.4524
#  gen: 293, stag: 6, mean: 7040.88, median: 8135.16, stdev: 3629.15, best: 707.4524
#  gen: 294, stag: 7, mean: 7352.83, median: 8227.39, stdev: 5545.23, best: 707.4524
#  gen: 295, stag: 8, mean: 10177.62, median: 10598.08, stdev: 7283.85, best: 707.4524
#  gen: 296, stag: 9, mean: 12970.47, median: 13676.25, stdev: 7277.18, best: 707.4524
#  gen: 297, stag: 10, mean: 13140.94, median: 13732.53, stdev: 7750.49, best: 707.4524
#  gen: 298, stag: 11, mean: 11073.49, median: 13284.36, stdev: 6968.67, best: 707.4524
#  gen: 299, stag: 12, mean: 9875.46, median: 12801.74, stdev: 7012.01, best: 707.4524
#  gen: 300, stag: 13, mean: 10317.86, median: 13284.36, stdev: 6051.38, best: 707.4524
#  gen: 301, stag: 14, mean: 12795.91, median: 15485.61, stdev: 6133.52, best: 707.4524

#  Total generations: 301
#  Total CPU time: 5.697144510999999 s
#  Best score: 707.4524479732933
#  Real chromosome of best solution: [ 0.18701042 -0.60048745  0.25149984 -1.11529317  1.64784057  0.94531226
#   2.26719708]
#  Binary chromosome of best solution: []
