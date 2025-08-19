import numpy as np

from evolvekit.core.Ga.GaIsland import GaIsland
from evolvekit.core.Ga.enums.GaClampStrategy import GaClampStrategy
from evolvekit.core.benchmarks import RosenbrockEvaluator

from evolvekit.operators.Ga.selection.RankSelection import RankSelection
from evolvekit.operators.Ga.crossover.real.OnePointCrossover import OnePointCrossover
from evolvekit.operators.Ga.mutation.real.VirusInfectionMutation import VirusInfectionMutation

from evolvekit.examples.inspectors.CSVInspector import CSVInspector


def optimize_rosenbrock() -> None:
    """
    Optimize the Rosenbrock function.

    Uses mutation patterns that gently bias genes towards 1.0 (the
    global optimum), while keeping enough randomness.
    """
    dim = 20
    evaluator = RosenbrockEvaluator(dim=dim)

    ga = GaIsland()
    ga.set_evaluator(evaluator)
    ga.set_inspector(
        CSVInspector(filename="rosenbrock_evolution.csv", stagnation_limit=200)
    )

    ga.set_population_size(300)
    ga.set_elite_count(30)

    ga.set_operator(RankSelection(target_population=30))
    ga.set_operator(OnePointCrossover())

    virus_vectors = []
    chunk = max(1, dim // 5)
    for i in range(5):
        pat = [None] * dim
        for j in range(i * chunk, min(dim, (i + 1) * chunk)):
            pat[j] = 1.0
        virus_vectors.append(pat)
        virus_vectors.append([1.0 + 0.05 * np.random.randn() for _ in range(dim)])

    ga.set_operator(
        VirusInfectionMutation(virus_vectors=virus_vectors, p_copy=0.35, p_replace=0.20)
    )

    ga.set_crossover_probability(0.80)
    ga.set_mutation_probability(0.10)
    ga.set_max_generations(900)
    ga.set_seed(12356)
    ga.set_real_clamp_strategy(GaClampStrategy.RANDOM)

    results = ga.run()
    logged_generations = results.total_generations

    print("Rosenbrock optimization")
    print(f"  best fitness: {results.value:.6f}")
    print(f"  generations:  {logged_generations}")
    print("  csv: rosenbrock_evolution.csv")
