import numpy as np

from evolvekit.core.Ga.GaIsland import GaIsland
from evolvekit.core.Ga.enums.GaClampStrategy import GaClampStrategy
from evolvekit.benchmarks import RastriginEvaluator

from evolvekit.operators.Ga.selection.RankSelection import RankSelection
from evolvekit.operators.Ga.crossover.real.OnePointCrossover import OnePointCrossover
from evolvekit.operators.Ga.mutation.real.VirusInfectionMutation import (
    VirusInfectionMutation,
)

from evolvekit.examples.inspectors.CSVInspector import CSVInspector


def optimize_rastrigin() -> None:
    """
    Optimize the Rastrigin function with real-valued GA.

    This setup is a bit more explorative than Sphere: slightly larger
    population and longer run.
    """
    dim = 15
    evaluator = RastriginEvaluator(dim=dim)

    ga = GaIsland()
    ga.set_evaluator(evaluator)
    ga.set_inspector(
        CSVInspector(filename="rastrigin_evolution.csv", stagnation_limit=150)
    )

    ga.set_population_size(200)
    ga.set_elite_count(15)

    ga.set_operator(RankSelection(target_population=15))
    ga.set_operator(OnePointCrossover())

    virus_vectors = [
        [0.0] * dim,  # full reset pattern
        [None if i < dim // 2 else 0.0 for i in range(dim)],  # half reset
        [
            np.random.normal(0.0, 0.7) if np.random.rand() < 0.5 else None
            for _ in range(dim)
        ],
    ]

    ga.set_operator(
        VirusInfectionMutation(
            virus_vectors=virus_vectors,
            p_copy=0.30,
            p_replace=0.15,
        )
    )

    ga.set_crossover_probability(0.85)
    ga.set_mutation_probability(0.15)
    ga.set_max_generations(600)
    ga.set_seed(2025323)
    ga.set_real_clamp_strategy(GaClampStrategy.CLAMP)

    results = ga.run()
    logged_generations = results.total_generations

    print("Rastrigin optimization")
    print(f"  best fitness: {results.value:.6f}")
    print(f"  generations:  {logged_generations}")
    print(f"  chromosome value: {results.real_chrom}")
    print("  csv: rastrigin_evolution.csv")
