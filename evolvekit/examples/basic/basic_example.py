from evolvekit.core.Ga.GaIsland import GaIsland
from evolvekit.core.benchmarks import RastriginEvaluator


def basic_example() -> None:
    """
    This is basic example to show how to use our library in the simplest way with default parameters and operators.

    The only thing you must provide is the evaluator! The remaining parameters and operators are set by default.
    You can create your own custom evaluator or use one from our benchmarks.
    """

    dim = 15
    evaluator = RastriginEvaluator(
        dim=dim
    )  # SphereEvaluator(dim=dim, bounds=(-5.12, 5.12))

    ga = GaIsland()
    ga.set_evaluator(evaluator)

    results = ga.run()
    logged_generations = results.total_generations

    print("Rastrigin optimization")
    print(f"  best fitness: {results.value:.6f}")
    print(f"  generations:  {logged_generations}")
