from evolvekit.core.Ga.GaIsland import GaIsland
from evolvekit.core.Ga.enums.GaClampStrategy import GaClampStrategy
from evolvekit.core.benchmarks import SphereEvaluator

from evolvekit.operators.Ga.selection.RankSelection import RankSelection
from evolvekit.operators.Ga.crossover.real.OnePointCrossover import OnePointCrossover
from evolvekit.operators.Ga.mutation.real.VirusInfectionMutation import VirusInfectionMutation

from evolvekit.examples.inspectors.CSVInspector import CSVInspector


def compare_clamp_strategies() -> None:
    """
    Compare different clamping strategies on the same Sphere task.

    The mutation deliberately pushes genes out of bounds so that the
    clamp strategy makes a visible difference between runs.
    """
    dim = 8
    evaluator = SphereEvaluator(dim=dim, bounds=(-5.12, 5.12))

    strategies = [
        GaClampStrategy.NONE,
        GaClampStrategy.CLAMP,
        GaClampStrategy.BOUNCE,
        GaClampStrategy.OVERFLOW,
        GaClampStrategy.RANDOM,
    ]

    for strategy in strategies:
        ga = GaIsland()
        ga.set_evaluator(evaluator)
        ga.set_inspector(
            CSVInspector(
                filename=f"strategy_{strategy.name.lower()}.csv",
                stagnation_limit=60,
            )
        )

        ga.set_population_size(80)
        ga.set_elite_count(8)

        ga.set_operator(RankSelection(target_population=8))
        ga.set_operator(OnePointCrossover())

        # Force out-of-bounds to highlight clamp behavior
        virus_vectors = [
            [9.0] * dim,   # above upper bound
            [-9.0] * dim,  # below lower bound
            [0.0 if i % 2 == 0 else None for i in range(dim)],
            ]
        ga.set_operator(
            VirusInfectionMutation(virus_vectors=virus_vectors, p_copy=0.4, p_replace=0.25)
        )

        ga.set_crossover_probability(0.85)
        ga.set_mutation_probability(0.25)
        ga.set_max_generations(400)
        ga.set_seed(12345999)  # same seed for fair comparison
        ga.set_real_clamp_strategy(strategy)

        results = ga.run()
        print(f"Clamp {strategy.name:<8} -> best fitness: {results.value:.6f}")