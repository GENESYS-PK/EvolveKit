from evolvekit.core.Ga.GaIsland import GaIsland
from evolvekit.core.Ga.enums.GaClampStrategy import GaClampStrategy
from evolvekit.core.benchmarks import SphereEvaluator

from evolvekit.operators.Ga.selection.RankSelection import RankSelection
from evolvekit.operators.Ga.crossover.real.OnePointCrossover import OnePointCrossover
from evolvekit.operators.Ga.mutation.real.VirusInfectionMutation import (
    VirusInfectionMutation,
)

from evolvekit.examples.inspectors.CSVInspector import CSVInspector


def optimize_sphere() -> None:
    """
    Optimize the Sphere function with real-valued GA.

    The mutation uses virus vectors that intentionally push genes
    **outside** the domain to demonstrate clamping.
    """
    dim = 10
    evaluator = SphereEvaluator(dim=dim, bounds=(-5.12, 5.12))

    ga = GaIsland()
    ga.set_evaluator(evaluator)

    inspector = CSVInspector(filename="sphere_evolution.csv", stagnation_limit=100)
    ga.set_inspector(inspector)

    ga.set_population_size(120)
    ga.set_elite_count(12)

    ga.set_operator(RankSelection(target_population=12))
    ga.set_operator(OnePointCrossover())

    # Out-of-domain mutation patterns to trigger clamp strategies
    virus_vectors = [
        [10.0] * dim,  # push all genes above the upper bound
        [-12.0] * dim,  # push all genes below the lower bound
        [0.0 if i % 3 == 0 else None for i in range(dim)],  # sparse reset to 0
    ]

    ga.set_operator(
        VirusInfectionMutation(
            virus_vectors=virus_vectors,
            p_copy=0.35,
            p_replace=0.25,
        )
    )

    ga.set_crossover_probability(0.85)
    ga.set_mutation_probability(0.25)
    ga.set_max_generations(400)
    ga.set_seed(78976)
    ga.set_real_clamp_strategy(GaClampStrategy.BOUNCE)

    results = ga.run()
    logged_generations = results.total_generations

    print("Sphere optimization")
    print(f"  best fitness: {results.value:.6f}")
    print(f"  generations:  {logged_generations}")
    print(f"  chromosome value: {results.real_chrom}")
    print("  csv: sphere_evolution.csv")
