from evolvekit.core.Ga.GaIsland import GaIsland
from evolvekit.core.benchmarks import RastriginEvaluator


def basic_example_with_simple_setup() -> None:
    """
    This is basic example to show how to use our library with custom setup some parameters.

    The only thing you must provide is the evaluator! The remaining parameters and operators are set by default.
    However, our library gives you the opportunity to custom configure every parameter! See how:
    """

    dim = 15
    evaluator = RastriginEvaluator(dim=dim)

    ga = GaIsland()
    ga.set_evaluator(evaluator)

    """
    You can customize all of the following parameters 
    """

    ga.set_crossover_probability(0.70)
    ga.set_mutation_probability(0.20)
    ga.set_max_generations(900)
    ga.set_seed(123456)
    ga.set_population_size(150)
    ga.set_elite_count(15)

    """
    You can also customize the crossover, mutation and selection operators.
    Additionally, you can add inspector which will allow you to monitor the course of the genetic algorithm.
    For more details look at the advanced example. 
    """

    results = ga.run()
    logged_generations = results.total_generations

    print("Rastrigin optimization")
    print(f"  best fitness: {results.value:.6f}")
    print(f"  generations:  {logged_generations}")
    print(f"  chromosome value: {results.real_chrom}")
