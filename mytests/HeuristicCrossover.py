from evolvekit import *
from evolvekit.operators.Ga.real.mutation.UniformMutation import UniformMutation
from evolvekit.operators.Ga.real.crossover.HeuristicCrossover import HeuristicCrossover


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
crossover_real = HeuristicCrossover()
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
#  gen: 2, stag: 1, mean: 81328.98, median: 72787.48, stdev: 58530.40, best: 6656.6039
#  gen: 3, stag: 2, mean: 84830.83, median: 77224.86, stdev: 58242.13, best: 6656.6039
#  gen: 4, stag: 3, mean: 104174.26, median: 95935.39, stdev: 74338.41, best: 6656.6039
#  gen: 5, stag: 4, mean: 96927.54, median: 94761.55, stdev: 74145.31, best: 6656.6039
#  gen: 6, stag: 5, mean: 116801.25, median: 103460.28, stdev: 84428.18, best: 6656.6039
#  gen: 7, stag: 6, mean: 115286.00, median: 98013.86, stdev: 78802.05, best: 6656.6039
#  gen: 8, stag: 7, mean: 130307.44, median: 135581.60, stdev: 85989.87, best: 6656.6039
#  gen: 9, stag: 8, mean: 116021.41, median: 114483.88, stdev: 81376.15, best: 6656.6039
#  gen: 10, stag: 9, mean: 136104.69, median: 129683.83, stdev: 88998.29, best: 6656.6039
#  gen: 11, stag: 10, mean: 129676.65, median: 124990.74, stdev: 83856.16, best: 6656.6039
#  gen: 12, stag: 11, mean: 142917.56, median: 141983.03, stdev: 91054.79, best: 6656.6039
#  gen: 13, stag: 12, mean: 110940.04, median: 103803.75, stdev: 85835.52, best: 6656.6039
#  gen: 14, stag: 13, mean: 93151.92, median: 75556.39, stdev: 77295.63, best: 6656.6039
#  gen: 15, stag: 14, mean: 117849.85, median: 118397.83, stdev: 74228.87, best: 6656.6039
#  gen: 16, stag: 15, mean: 109251.07, median: 106858.15, stdev: 78870.32, best: 6656.6039
#  gen: 17, stag: 16, mean: 113438.38, median: 118276.11, stdev: 81557.28, best: 6656.6039
#  gen: 18, stag: 0, mean: 101488.85, median: 75461.09, stdev: 81767.40, best: 2991.8630
#  gen: 19, stag: 1, mean: 93891.80, median: 83706.70, stdev: 66782.78, best: 2991.8630
#  gen: 20, stag: 2, mean: 98039.98, median: 90335.31, stdev: 75422.07, best: 2991.8630
#  gen: 21, stag: 3, mean: 112821.26, median: 127634.57, stdev: 80566.81, best: 2991.8630
#  gen: 22, stag: 4, mean: 100441.12, median: 98532.59, stdev: 76126.61, best: 2991.8630
#  gen: 23, stag: 5, mean: 125595.63, median: 134264.46, stdev: 86984.85, best: 2991.8630
#  gen: 24, stag: 6, mean: 93032.16, median: 81752.73, stdev: 79776.85, best: 2991.8630
#  gen: 25, stag: 7, mean: 98339.42, median: 100074.86, stdev: 75266.35, best: 2991.8630
#  gen: 26, stag: 8, mean: 131032.78, median: 123094.52, stdev: 93714.28, best: 2991.8630
#  gen: 27, stag: 9, mean: 141087.12, median: 135747.35, stdev: 90974.45, best: 2991.8630
#  gen: 28, stag: 10, mean: 116945.80, median: 114890.58, stdev: 84571.20, best: 2991.8630
#  gen: 29, stag: 11, mean: 102548.52, median: 102452.71, stdev: 74204.39, best: 2991.8630
#  gen: 30, stag: 12, mean: 109334.88, median: 105869.81, stdev: 76841.11, best: 2991.8630
#  gen: 31, stag: 13, mean: 104667.09, median: 103945.23, stdev: 78141.33, best: 2991.8630
#  gen: 32, stag: 14, mean: 108402.20, median: 105047.24, stdev: 90510.88, best: 2991.8630
#  gen: 33, stag: 15, mean: 134927.31, median: 157101.45, stdev: 91154.79, best: 2991.8630
#  gen: 34, stag: 16, mean: 105143.21, median: 100521.36, stdev: 84705.18, best: 2991.8630
#  gen: 35, stag: 17, mean: 96791.48, median: 113332.05, stdev: 69067.72, best: 2991.8630
#  gen: 36, stag: 18, mean: 107147.57, median: 100778.95, stdev: 81427.08, best: 2991.8630
#  gen: 37, stag: 19, mean: 102053.48, median: 97406.65, stdev: 75540.17, best: 2991.8630
#  gen: 38, stag: 20, mean: 100009.45, median: 98596.43, stdev: 73191.07, best: 2991.8630
#  gen: 39, stag: 21, mean: 108378.68, median: 109362.17, stdev: 81683.21, best: 2991.8630
#  gen: 40, stag: 22, mean: 124489.79, median: 125603.54, stdev: 93730.23, best: 2991.8630
#  gen: 41, stag: 0, mean: 111284.54, median: 106402.29, stdev: 91293.24, best: 2109.9893
#  gen: 42, stag: 1, mean: 104371.75, median: 106104.44, stdev: 74790.01, best: 2109.9893
#  gen: 43, stag: 2, mean: 99296.83, median: 91930.12, stdev: 81507.69, best: 2109.9893
#  gen: 44, stag: 3, mean: 120695.70, median: 120332.69, stdev: 93838.88, best: 2109.9893
#  gen: 45, stag: 4, mean: 145851.86, median: 144711.49, stdev: 102880.76, best: 2109.9893
#  gen: 46, stag: 5, mean: 139540.88, median: 152424.56, stdev: 96433.65, best: 2109.9893
#  gen: 47, stag: 6, mean: 156893.73, median: 189564.19, stdev: 92248.86, best: 2109.9893
#  gen: 48, stag: 7, mean: 139281.51, median: 143288.34, stdev: 99163.11, best: 2109.9893
#  gen: 49, stag: 8, mean: 105000.76, median: 89363.44, stdev: 88665.38, best: 2109.9893
#  gen: 50, stag: 9, mean: 95934.10, median: 82673.64, stdev: 78645.33, best: 2109.9893
#  gen: 51, stag: 10, mean: 96309.89, median: 75376.14, stdev: 81153.00, best: 2109.9893
#  gen: 52, stag: 11, mean: 101520.32, median: 93905.93, stdev: 77135.19, best: 2109.9893
#  gen: 53, stag: 12, mean: 94484.36, median: 88932.30, stdev: 72671.50, best: 2109.9893
#  gen: 54, stag: 13, mean: 88712.55, median: 73138.32, stdev: 76998.77, best: 2109.9893
#  gen: 55, stag: 14, mean: 78169.20, median: 61012.12, stdev: 65940.30, best: 2109.9893
#  gen: 56, stag: 15, mean: 98877.09, median: 101225.57, stdev: 72350.55, best: 2109.9893
#  gen: 57, stag: 16, mean: 101382.57, median: 99260.26, stdev: 77404.24, best: 2109.9893
#  gen: 58, stag: 17, mean: 100831.17, median: 83443.75, stdev: 77556.97, best: 2109.9893
#  gen: 59, stag: 18, mean: 120849.82, median: 131314.35, stdev: 82256.32, best: 2109.9893
#  gen: 60, stag: 19, mean: 108570.49, median: 107910.09, stdev: 78336.37, best: 2109.9893
#  gen: 61, stag: 20, mean: 106718.32, median: 122728.41, stdev: 84377.77, best: 2109.9893
#  gen: 62, stag: 21, mean: 96979.60, median: 83276.08, stdev: 71808.10, best: 2109.9893
#  gen: 63, stag: 22, mean: 102235.88, median: 99562.44, stdev: 70602.62, best: 2109.9893
#  gen: 64, stag: 23, mean: 100026.60, median: 99326.43, stdev: 71078.49, best: 2109.9893
#  gen: 65, stag: 24, mean: 112531.25, median: 137459.59, stdev: 75340.35, best: 2109.9893
#  gen: 66, stag: 25, mean: 102707.62, median: 88228.21, stdev: 80333.59, best: 2109.9893
#  gen: 67, stag: 26, mean: 107848.63, median: 99927.09, stdev: 79044.80, best: 2109.9893
#  gen: 68, stag: 27, mean: 102949.81, median: 93477.80, stdev: 78012.58, best: 2109.9893
#  gen: 69, stag: 28, mean: 88591.66, median: 85940.27, stdev: 65200.62, best: 2109.9893
#  gen: 70, stag: 29, mean: 103571.40, median: 104979.66, stdev: 85750.77, best: 2109.9893
#  gen: 71, stag: 30, mean: 119108.63, median: 116972.66, stdev: 94326.90, best: 2109.9893
#  gen: 72, stag: 31, mean: 96879.93, median: 99629.91, stdev: 73723.38, best: 2109.9893
#  gen: 73, stag: 32, mean: 99077.85, median: 103455.03, stdev: 73803.52, best: 2109.9893
#  gen: 74, stag: 33, mean: 113972.47, median: 127169.02, stdev: 70808.11, best: 2109.9893
#  gen: 75, stag: 34, mean: 123204.53, median: 134640.72, stdev: 83248.45, best: 2109.9893
#  gen: 76, stag: 35, mean: 141672.27, median: 166918.66, stdev: 88554.44, best: 2109.9893
#  gen: 77, stag: 36, mean: 101649.70, median: 86394.80, stdev: 81660.86, best: 2109.9893
#  gen: 78, stag: 37, mean: 76032.34, median: 69893.02, stdev: 58507.78, best: 2109.9893
#  gen: 79, stag: 38, mean: 66712.23, median: 63789.81, stdev: 51697.85, best: 2109.9893
#  gen: 80, stag: 39, mean: 82018.78, median: 79159.15, stdev: 64687.99, best: 2109.9893
#  gen: 81, stag: 40, mean: 98326.99, median: 82216.12, stdev: 83636.61, best: 2109.9893
#  gen: 82, stag: 41, mean: 143622.16, median: 153074.14, stdev: 107904.94, best: 2109.9893
#  gen: 83, stag: 42, mean: 160195.91, median: 157954.96, stdev: 114261.93, best: 2109.9893
#  gen: 84, stag: 43, mean: 156912.82, median: 162549.34, stdev: 109684.65, best: 2109.9893
#  gen: 85, stag: 44, mean: 149042.80, median: 150315.26, stdev: 107639.10, best: 2109.9893
#  gen: 86, stag: 45, mean: 99407.42, median: 80950.09, stdev: 85943.92, best: 2109.9893
#  gen: 87, stag: 46, mean: 82347.85, median: 59110.43, stdev: 76361.63, best: 2109.9893
#  gen: 88, stag: 47, mean: 110037.50, median: 109629.22, stdev: 81509.81, best: 2109.9893
#  gen: 89, stag: 48, mean: 96972.29, median: 82768.16, stdev: 78879.35, best: 2109.9893
#  gen: 90, stag: 49, mean: 92854.64, median: 84885.57, stdev: 78080.68, best: 2109.9893
#  gen: 91, stag: 50, mean: 96259.49, median: 98888.24, stdev: 68505.13, best: 2109.9893
#  gen: 92, stag: 51, mean: 120907.60, median: 138688.42, stdev: 80115.60, best: 2109.9893
#  gen: 93, stag: 52, mean: 128365.48, median: 125354.14, stdev: 92207.39, best: 2109.9893
#  gen: 94, stag: 53, mean: 127020.53, median: 134651.73, stdev: 89651.15, best: 2109.9893
#  gen: 95, stag: 54, mean: 128252.82, median: 123856.18, stdev: 93703.48, best: 2109.9893
#  gen: 96, stag: 55, mean: 125782.72, median: 126019.20, stdev: 85471.53, best: 2109.9893
#  gen: 97, stag: 56, mean: 113804.53, median: 119307.05, stdev: 80661.76, best: 2109.9893
#  gen: 98, stag: 57, mean: 104049.18, median: 99303.64, stdev: 85520.01, best: 2109.9893
#  gen: 99, stag: 58, mean: 116629.57, median: 121750.54, stdev: 75572.62, best: 2109.9893
#  gen: 100, stag: 0, mean: 123207.53, median: 129477.43, stdev: 83291.59, best: 1984.9202
#  gen: 101, stag: 1, mean: 126273.13, median: 130761.93, stdev: 96881.47, best: 1984.9202
#  gen: 102, stag: 2, mean: 137091.74, median: 130879.37, stdev: 98844.61, best: 1984.9202
#  gen: 103, stag: 3, mean: 100573.75, median: 90193.25, stdev: 79592.62, best: 1984.9202
#  gen: 104, stag: 4, mean: 88170.02, median: 78408.93, stdev: 70710.90, best: 1984.9202
#  gen: 105, stag: 5, mean: 95826.16, median: 85198.27, stdev: 73007.71, best: 1984.9202
#  gen: 106, stag: 6, mean: 103531.51, median: 85198.27, stdev: 81602.85, best: 1984.9202
#  gen: 107, stag: 7, mean: 118813.36, median: 126909.73, stdev: 84045.24, best: 1984.9202
#  gen: 108, stag: 8, mean: 131244.64, median: 150141.14, stdev: 89627.66, best: 1984.9202
#  gen: 109, stag: 9, mean: 123605.65, median: 126698.03, stdev: 89125.15, best: 1984.9202
#  gen: 110, stag: 10, mean: 111346.05, median: 108688.46, stdev: 85791.92, best: 1984.9202
#  gen: 111, stag: 11, mean: 86563.25, median: 75484.51, stdev: 69813.82, best: 1984.9202
#  gen: 112, stag: 12, mean: 113617.22, median: 116118.64, stdev: 81048.86, best: 1984.9202
#  gen: 113, stag: 13, mean: 113114.72, median: 127424.68, stdev: 78732.10, best: 1984.9202
#  gen: 114, stag: 14, mean: 118894.48, median: 120637.43, stdev: 89922.87, best: 1984.9202
#  gen: 115, stag: 15, mean: 88264.13, median: 76240.79, stdev: 70144.46, best: 1984.9202
#  gen: 116, stag: 16, mean: 110021.30, median: 120918.97, stdev: 81208.77, best: 1984.9202
#  gen: 117, stag: 17, mean: 117535.08, median: 118096.51, stdev: 87921.62, best: 1984.9202
#  gen: 118, stag: 18, mean: 134018.54, median: 127232.96, stdev: 90986.70, best: 1984.9202
#  gen: 119, stag: 19, mean: 156643.79, median: 176852.91, stdev: 94235.82, best: 1984.9202
#  gen: 120, stag: 20, mean: 117863.23, median: 123644.15, stdev: 97813.00, best: 1984.9202
#  gen: 121, stag: 21, mean: 97902.37, median: 50899.52, stdev: 91931.91, best: 1984.9202
#  gen: 122, stag: 22, mean: 148664.37, median: 181985.24, stdev: 92388.32, best: 1984.9202
#  gen: 123, stag: 23, mean: 136134.22, median: 149407.75, stdev: 88754.85, best: 1984.9202
#  gen: 124, stag: 24, mean: 136405.03, median: 136680.23, stdev: 95684.27, best: 1984.9202
#  gen: 125, stag: 25, mean: 115858.92, median: 102174.82, stdev: 95111.86, best: 1984.9202
#  gen: 126, stag: 26, mean: 118139.98, median: 118482.97, stdev: 89517.32, best: 1984.9202
#  gen: 127, stag: 27, mean: 96718.04, median: 96671.79, stdev: 74314.61, best: 1984.9202
#  gen: 128, stag: 28, mean: 92366.66, median: 85242.27, stdev: 70497.24, best: 1984.9202
#  gen: 129, stag: 29, mean: 110960.02, median: 99479.17, stdev: 86236.44, best: 1984.9202
#  gen: 130, stag: 30, mean: 123904.86, median: 118936.48, stdev: 88046.40, best: 1984.9202
#  gen: 131, stag: 31, mean: 100716.12, median: 95863.39, stdev: 73274.82, best: 1984.9202
#  gen: 132, stag: 32, mean: 81005.92, median: 79002.07, stdev: 56281.02, best: 1984.9202
#  gen: 133, stag: 33, mean: 112243.78, median: 107777.59, stdev: 80041.92, best: 1984.9202
#  gen: 134, stag: 34, mean: 137433.74, median: 149183.84, stdev: 91434.93, best: 1984.9202
#  gen: 135, stag: 35, mean: 142018.69, median: 170635.08, stdev: 93353.43, best: 1984.9202
#  gen: 136, stag: 36, mean: 142224.66, median: 159670.16, stdev: 93720.42, best: 1984.9202
#  gen: 137, stag: 37, mean: 118564.69, median: 121039.24, stdev: 88531.05, best: 1984.9202
#  gen: 138, stag: 38, mean: 123764.14, median: 128397.40, stdev: 91029.12, best: 1984.9202
#  gen: 139, stag: 39, mean: 97179.52, median: 85130.87, stdev: 78237.92, best: 1984.9202
#  gen: 140, stag: 40, mean: 122670.34, median: 137048.77, stdev: 90415.14, best: 1984.9202
#  gen: 141, stag: 41, mean: 150331.23, median: 175726.64, stdev: 95315.44, best: 1984.9202
#  gen: 142, stag: 42, mean: 160133.13, median: 189870.54, stdev: 100069.37, best: 1984.9202
#  gen: 143, stag: 43, mean: 153207.92, median: 178833.88, stdev: 99093.85, best: 1984.9202
#  gen: 144, stag: 44, mean: 142691.22, median: 176430.70, stdev: 97927.36, best: 1984.9202
#  gen: 145, stag: 45, mean: 161324.90, median: 193509.61, stdev: 100416.88, best: 1984.9202
#  gen: 146, stag: 46, mean: 144013.66, median: 155249.22, stdev: 96909.32, best: 1984.9202
#  gen: 147, stag: 47, mean: 121575.23, median: 117510.58, stdev: 91644.26, best: 1984.9202
#  gen: 148, stag: 48, mean: 103532.09, median: 85799.26, stdev: 87841.98, best: 1984.9202
#  gen: 149, stag: 49, mean: 126623.52, median: 105698.18, stdev: 96472.97, best: 1984.9202
#  gen: 150, stag: 50, mean: 103276.66, median: 95601.75, stdev: 80441.83, best: 1984.9202
#  gen: 151, stag: 51, mean: 90977.63, median: 87675.11, stdev: 72261.44, best: 1984.9202
#  gen: 152, stag: 52, mean: 97106.95, median: 94273.38, stdev: 69931.29, best: 1984.9202
#  gen: 153, stag: 53, mean: 92865.07, median: 91923.33, stdev: 68095.85, best: 1984.9202
#  gen: 154, stag: 54, mean: 103183.96, median: 105260.51, stdev: 73914.81, best: 1984.9202
#  gen: 155, stag: 55, mean: 131802.32, median: 142582.77, stdev: 80379.04, best: 1984.9202
#  gen: 156, stag: 56, mean: 118465.47, median: 134820.94, stdev: 81213.81, best: 1984.9202
#  gen: 157, stag: 57, mean: 121011.45, median: 137377.78, stdev: 79477.25, best: 1984.9202
#  gen: 158, stag: 58, mean: 122619.93, median: 119887.93, stdev: 93052.11, best: 1984.9202
#  gen: 159, stag: 59, mean: 126034.77, median: 131364.29, stdev: 89407.79, best: 1984.9202
#  gen: 160, stag: 60, mean: 126926.34, median: 156214.23, stdev: 90015.53, best: 1984.9202
#  gen: 161, stag: 61, mean: 136183.61, median: 151929.27, stdev: 97563.73, best: 1984.9202
#  gen: 162, stag: 62, mean: 178076.85, median: 172229.06, stdev: 116015.69, best: 1984.9202
#  gen: 163, stag: 63, mean: 145183.75, median: 153215.74, stdev: 104273.18, best: 1984.9202
#  gen: 164, stag: 64, mean: 123574.96, median: 130299.25, stdev: 85563.13, best: 1984.9202
#  gen: 165, stag: 65, mean: 129430.18, median: 128128.51, stdev: 91127.40, best: 1984.9202
#  gen: 166, stag: 0, mean: 104374.92, median: 117282.25, stdev: 85113.37, best: 543.8494
#  gen: 167, stag: 1, mean: 89697.27, median: 85114.79, stdev: 69156.37, best: 543.8494
#  gen: 168, stag: 2, mean: 96808.74, median: 96773.24, stdev: 73740.22, best: 543.8494
#  gen: 169, stag: 3, mean: 73653.74, median: 56907.28, stdev: 65385.72, best: 543.8494
#  gen: 170, stag: 4, mean: 91054.67, median: 96985.23, stdev: 67625.84, best: 543.8494
#  gen: 171, stag: 5, mean: 105386.58, median: 118879.37, stdev: 68637.24, best: 543.8494
#  gen: 172, stag: 6, mean: 105670.88, median: 111555.24, stdev: 74259.82, best: 543.8494
#  gen: 173, stag: 7, mean: 114695.28, median: 132985.98, stdev: 74624.87, best: 543.8494
#  gen: 174, stag: 8, mean: 126832.53, median: 141793.17, stdev: 76362.16, best: 543.8494
#  gen: 175, stag: 9, mean: 123439.98, median: 111264.02, stdev: 91753.97, best: 543.8494
#  gen: 176, stag: 10, mean: 119123.77, median: 116574.60, stdev: 88142.02, best: 543.8494
#  gen: 177, stag: 11, mean: 136335.53, median: 145522.09, stdev: 89043.98, best: 543.8494
#  gen: 178, stag: 12, mean: 120413.42, median: 120462.40, stdev: 86032.46, best: 543.8494
#  gen: 179, stag: 13, mean: 102983.02, median: 104130.70, stdev: 82472.72, best: 543.8494
#  gen: 180, stag: 14, mean: 75590.64, median: 59079.55, stdev: 63275.48, best: 543.8494
#  gen: 181, stag: 15, mean: 89085.97, median: 79804.89, stdev: 70069.34, best: 543.8494
#  gen: 182, stag: 16, mean: 110357.44, median: 104087.17, stdev: 86895.30, best: 543.8494
#  gen: 183, stag: 17, mean: 147488.10, median: 157395.27, stdev: 105664.27, best: 543.8494
#  gen: 184, stag: 18, mean: 94896.09, median: 69089.73, stdev: 89865.23, best: 543.8494
#  gen: 185, stag: 19, mean: 113521.93, median: 82125.08, stdev: 102384.22, best: 543.8494
#  gen: 186, stag: 20, mean: 132618.51, median: 122170.54, stdev: 104385.14, best: 543.8494
#  gen: 187, stag: 21, mean: 141301.92, median: 161043.11, stdev: 106236.13, best: 543.8494
#  gen: 188, stag: 22, mean: 127776.31, median: 132080.80, stdev: 93401.03, best: 543.8494
#  gen: 189, stag: 23, mean: 130376.28, median: 132853.86, stdev: 91596.72, best: 543.8494
#  gen: 190, stag: 24, mean: 103931.24, median: 108593.35, stdev: 77793.22, best: 543.8494
#  gen: 191, stag: 25, mean: 113273.09, median: 112821.71, stdev: 78821.53, best: 543.8494
#  gen: 192, stag: 26, mean: 132985.36, median: 144783.23, stdev: 94280.90, best: 543.8494
#  gen: 193, stag: 27, mean: 98758.42, median: 74339.36, stdev: 102763.14, best: 543.8494
#  gen: 194, stag: 28, mean: 86412.66, median: 88709.59, stdev: 66108.05, best: 543.8494
#  gen: 195, stag: 29, mean: 72996.94, median: 53790.13, stdev: 66139.53, best: 543.8494
#  gen: 196, stag: 30, mean: 93677.90, median: 92947.36, stdev: 65982.44, best: 543.8494
#  gen: 197, stag: 31, mean: 83734.60, median: 75173.27, stdev: 68956.02, best: 543.8494
#  gen: 198, stag: 32, mean: 99029.72, median: 92171.74, stdev: 73798.89, best: 543.8494
#  gen: 199, stag: 33, mean: 108578.31, median: 102814.44, stdev: 79940.92, best: 543.8494
#  gen: 200, stag: 34, mean: 122907.20, median: 122180.16, stdev: 82495.31, best: 543.8494
#  gen: 201, stag: 35, mean: 91066.38, median: 91638.40, stdev: 70464.12, best: 543.8494
#  gen: 202, stag: 36, mean: 116468.75, median: 128129.52, stdev: 80606.34, best: 543.8494
#  gen: 203, stag: 37, mean: 139147.24, median: 158679.36, stdev: 88733.72, best: 543.8494
#  gen: 204, stag: 38, mean: 139821.01, median: 154694.07, stdev: 87453.08, best: 543.8494
#  gen: 205, stag: 39, mean: 137211.55, median: 132185.62, stdev: 97525.88, best: 543.8494
#  gen: 206, stag: 40, mean: 132113.30, median: 110427.77, stdev: 96653.14, best: 543.8494
#  gen: 207, stag: 41, mean: 125853.89, median: 121092.53, stdev: 83171.84, best: 543.8494
#  gen: 208, stag: 42, mean: 129415.87, median: 152267.44, stdev: 91455.55, best: 543.8494
#  gen: 209, stag: 43, mean: 123960.00, median: 114304.78, stdev: 87658.17, best: 543.8494
#  gen: 210, stag: 44, mean: 121779.75, median: 109194.11, stdev: 99236.44, best: 543.8494
#  gen: 211, stag: 45, mean: 120078.87, median: 116374.88, stdev: 94234.29, best: 543.8494
#  gen: 212, stag: 46, mean: 92802.50, median: 81736.73, stdev: 70730.99, best: 543.8494
#  gen: 213, stag: 47, mean: 102188.20, median: 104074.46, stdev: 75069.32, best: 543.8494
#  gen: 214, stag: 48, mean: 107245.12, median: 105571.23, stdev: 77915.08, best: 543.8494
#  gen: 215, stag: 49, mean: 109514.88, median: 109382.83, stdev: 81009.63, best: 543.8494
#  gen: 216, stag: 50, mean: 144490.72, median: 175148.35, stdev: 101758.25, best: 543.8494
#  gen: 217, stag: 51, mean: 134609.87, median: 144674.83, stdev: 94707.17, best: 543.8494
#  gen: 218, stag: 52, mean: 99206.94, median: 89571.26, stdev: 82070.70, best: 543.8494
#  gen: 219, stag: 53, mean: 101855.26, median: 98852.56, stdev: 73748.43, best: 543.8494
#  gen: 220, stag: 54, mean: 101587.42, median: 78958.76, stdev: 80235.34, best: 543.8494
#  gen: 221, stag: 55, mean: 110623.19, median: 93805.53, stdev: 80188.27, best: 543.8494
#  gen: 222, stag: 56, mean: 96164.45, median: 89734.39, stdev: 83168.36, best: 543.8494
#  gen: 223, stag: 57, mean: 76652.00, median: 63129.95, stdev: 66643.85, best: 543.8494
#  gen: 224, stag: 58, mean: 84777.46, median: 70348.53, stdev: 68892.51, best: 543.8494
#  gen: 225, stag: 59, mean: 93102.14, median: 95736.33, stdev: 65886.51, best: 543.8494
#  gen: 226, stag: 60, mean: 81559.86, median: 80196.82, stdev: 59655.68, best: 543.8494
#  gen: 227, stag: 61, mean: 96158.50, median: 90691.40, stdev: 73989.95, best: 543.8494
#  gen: 228, stag: 62, mean: 86658.89, median: 72251.42, stdev: 69243.28, best: 543.8494
#  gen: 229, stag: 63, mean: 83965.49, median: 81097.65, stdev: 66460.80, best: 543.8494
#  gen: 230, stag: 64, mean: 80078.47, median: 71414.62, stdev: 70306.57, best: 543.8494
#  gen: 231, stag: 65, mean: 74208.49, median: 60052.68, stdev: 62543.57, best: 543.8494
#  gen: 232, stag: 66, mean: 69767.92, median: 53982.71, stdev: 58857.17, best: 543.8494
#  gen: 233, stag: 67, mean: 99213.90, median: 96756.38, stdev: 73864.03, best: 543.8494
#  gen: 234, stag: 68, mean: 127387.12, median: 134742.07, stdev: 89088.11, best: 543.8494
#  gen: 235, stag: 69, mean: 98789.55, median: 90959.83, stdev: 81852.58, best: 543.8494
#  gen: 236, stag: 70, mean: 99198.35, median: 116745.80, stdev: 69689.00, best: 543.8494
#  gen: 237, stag: 71, mean: 116952.65, median: 133530.86, stdev: 78773.05, best: 543.8494
#  gen: 238, stag: 72, mean: 122152.39, median: 131181.83, stdev: 86054.28, best: 543.8494
#  gen: 239, stag: 73, mean: 114163.11, median: 119784.84, stdev: 80472.09, best: 543.8494
#  gen: 240, stag: 74, mean: 108284.82, median: 101691.12, stdev: 81476.57, best: 543.8494
#  gen: 241, stag: 75, mean: 107284.97, median: 99688.77, stdev: 77883.49, best: 543.8494
#  gen: 242, stag: 76, mean: 119144.45, median: 116684.18, stdev: 81504.02, best: 543.8494
#  gen: 243, stag: 77, mean: 114265.93, median: 118413.34, stdev: 82824.54, best: 543.8494
#  gen: 244, stag: 78, mean: 141280.68, median: 160329.64, stdev: 102926.57, best: 543.8494
#  gen: 245, stag: 79, mean: 163473.31, median: 192006.71, stdev: 92518.97, best: 543.8494
#  gen: 246, stag: 80, mean: 147220.53, median: 165575.93, stdev: 92787.32, best: 543.8494
#  gen: 247, stag: 81, mean: 172183.34, median: 197671.57, stdev: 97058.96, best: 543.8494
#  gen: 248, stag: 82, mean: 171716.69, median: 196474.58, stdev: 96315.00, best: 543.8494
#  gen: 249, stag: 83, mean: 177147.38, median: 211797.43, stdev: 98497.03, best: 543.8494
#  gen: 250, stag: 84, mean: 143535.00, median: 164766.11, stdev: 106050.43, best: 543.8494
#  gen: 251, stag: 85, mean: 137399.94, median: 156373.43, stdev: 86633.57, best: 543.8494
#  gen: 252, stag: 86, mean: 118026.20, median: 113079.83, stdev: 100889.10, best: 543.8494
#  gen: 253, stag: 87, mean: 100102.86, median: 82313.30, stdev: 82755.62, best: 543.8494
#  gen: 254, stag: 88, mean: 108636.00, median: 109453.03, stdev: 75586.68, best: 543.8494
#  gen: 255, stag: 89, mean: 69994.16, median: 59848.49, stdev: 63528.60, best: 543.8494
#  gen: 256, stag: 90, mean: 94056.36, median: 107335.98, stdev: 65287.16, best: 543.8494
#  gen: 257, stag: 91, mean: 92459.69, median: 87553.24, stdev: 66712.03, best: 543.8494
#  gen: 258, stag: 92, mean: 88062.23, median: 75027.41, stdev: 69533.95, best: 543.8494
#  gen: 259, stag: 93, mean: 93809.31, median: 81828.65, stdev: 73481.76, best: 543.8494
#  gen: 260, stag: 94, mean: 112181.14, median: 96452.35, stdev: 90864.94, best: 543.8494
#  gen: 261, stag: 95, mean: 116346.69, median: 86823.68, stdev: 110100.98, best: 543.8494
#  gen: 262, stag: 96, mean: 85081.15, median: 75263.34, stdev: 72259.98, best: 543.8494
#  gen: 263, stag: 97, mean: 117684.18, median: 130070.82, stdev: 79632.19, best: 543.8494
#  gen: 264, stag: 98, mean: 128566.73, median: 138198.88, stdev: 83332.08, best: 543.8494
#  gen: 265, stag: 99, mean: 137719.01, median: 150383.84, stdev: 92557.92, best: 543.8494
#  gen: 266, stag: 100, mean: 135025.76, median: 160497.15, stdev: 97521.11, best: 543.8494
#  gen: 267, stag: 101, mean: 125158.08, median: 106850.72, stdev: 105641.82, best: 543.8494
#  gen: 268, stag: 102, mean: 115242.44, median: 95959.79, stdev: 89807.90, best: 543.8494
#  gen: 269, stag: 103, mean: 107545.75, median: 98419.65, stdev: 87484.12, best: 543.8494
#  gen: 270, stag: 104, mean: 102216.21, median: 95195.90, stdev: 80829.30, best: 543.8494
#  gen: 271, stag: 105, mean: 83521.31, median: 76132.90, stdev: 71223.40, best: 543.8494
#  gen: 272, stag: 106, mean: 85138.03, median: 82976.15, stdev: 70778.85, best: 543.8494
#  gen: 273, stag: 107, mean: 104320.72, median: 98137.43, stdev: 76768.72, best: 543.8494
#  gen: 274, stag: 108, mean: 121518.78, median: 131758.64, stdev: 87165.56, best: 543.8494
#  gen: 275, stag: 109, mean: 159387.53, median: 177062.30, stdev: 91132.43, best: 543.8494
#  gen: 276, stag: 110, mean: 153304.55, median: 177876.79, stdev: 91956.32, best: 543.8494
#  gen: 277, stag: 111, mean: 135163.57, median: 162292.30, stdev: 88522.39, best: 543.8494
#  gen: 278, stag: 112, mean: 118399.75, median: 96084.53, stdev: 89528.78, best: 543.8494
#  gen: 279, stag: 113, mean: 76774.63, median: 61455.95, stdev: 70192.63, best: 543.8494
#  gen: 280, stag: 114, mean: 95694.89, median: 78837.00, stdev: 75649.12, best: 543.8494
#  gen: 281, stag: 115, mean: 85219.14, median: 85980.90, stdev: 69411.42, best: 543.8494
#  gen: 282, stag: 116, mean: 96102.08, median: 87971.34, stdev: 71932.66, best: 543.8494
#  gen: 283, stag: 117, mean: 122547.38, median: 124219.73, stdev: 91522.77, best: 543.8494
#  gen: 284, stag: 118, mean: 100060.77, median: 104008.45, stdev: 78469.55, best: 543.8494
#  gen: 285, stag: 119, mean: 109672.97, median: 94661.48, stdev: 92375.77, best: 543.8494
#  gen: 286, stag: 120, mean: 95480.08, median: 77579.65, stdev: 81247.41, best: 543.8494
#  gen: 287, stag: 121, mean: 100794.93, median: 86898.59, stdev: 80725.70, best: 543.8494
#  gen: 288, stag: 122, mean: 115318.04, median: 111633.57, stdev: 81534.12, best: 543.8494
#  gen: 289, stag: 123, mean: 107666.94, median: 115134.09, stdev: 79182.60, best: 543.8494
#  gen: 290, stag: 124, mean: 107813.38, median: 120521.05, stdev: 82551.16, best: 543.8494
#  gen: 291, stag: 125, mean: 128071.31, median: 143506.64, stdev: 87822.79, best: 543.8494
#  gen: 292, stag: 126, mean: 130686.76, median: 137228.13, stdev: 95948.15, best: 543.8494
#  gen: 293, stag: 127, mean: 112665.22, median: 101579.10, stdev: 97170.60, best: 543.8494
#  gen: 294, stag: 128, mean: 97738.72, median: 75576.04, stdev: 83865.54, best: 543.8494
#  gen: 295, stag: 129, mean: 90932.51, median: 86596.27, stdev: 70055.94, best: 543.8494
#  gen: 296, stag: 130, mean: 89618.93, median: 84053.96, stdev: 71931.55, best: 543.8494
#  gen: 297, stag: 131, mean: 104064.51, median: 107706.38, stdev: 79798.46, best: 543.8494
#  gen: 298, stag: 132, mean: 101362.28, median: 111664.90, stdev: 80089.03, best: 543.8494
#  gen: 299, stag: 133, mean: 108312.13, median: 110295.91, stdev: 81696.21, best: 543.8494
#  gen: 300, stag: 134, mean: 107161.36, median: 113637.80, stdev: 79715.66, best: 543.8494
#  gen: 301, stag: 135, mean: 103763.86, median: 109544.02, stdev: 74995.01, best: 543.8494

#  Total generations: 301
#  Total CPU time: 8.814431411000001 s
#  Best score: 543.8493759861841
#  Real chromosome of best solution: [-0.94964962  0.78070774 -0.53017759  1.07671717  0.90461532  2.62557818
#   6.62455989]
#  Binary chromosome of best solution: []
