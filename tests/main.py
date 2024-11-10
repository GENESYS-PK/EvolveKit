from core.evolution import EvolutionBuilder
from core.evolution import Evolution
from Options import MUTATION, CROSSOVER
from core.impl.operators.selections import TournamentSelection
from core.fitness_function import FitnessFunction
from core.impl.clamps import SimpleClampStrategy
import numpy as np
from core import Representation
from core import Population
from core import Individual
from typing import List, Tuple


def sample_fitness_function(chromosome: np.ndarray) -> float:
    return np.sum(chromosome)


def custom_population_generator(
    population_size: int, individual_size: int, variable_domains: List[Tuple[int, int]]
) -> Population:
    """
    Generates a population of individuals with random values within specified domains.

    :param population_size: The size of the population to generate.
    :param individual_size: The size of each individual (number of variables).
    :param variable_domains: A list of tuples specifying the min and max for each variable.
    :return: A Population instance containing the generated individuals.
    """
    individuals = []

    for _ in range(population_size):
        # Create a random individual
        individual_genes = []

        for domain in variable_domains:
            min_value, max_value = domain
            # Generate a random value within the specified domain
            gene_value = np.random.uniform(min_value, max_value)
            individual_genes.append(gene_value)

        # Assuming the Individual class can be initialized with a list of genes
        individual = Individual(individual_genes, gene_value)
        individuals.append(individual)
    return Population(population=individuals)


def main():
    evoBuilder = EvolutionBuilder(Evolution)
    evoBuilder.set_population_size(10).set_maximize(False)
    evoBuilder.set_population_generator(
        custom_population_generator, evoBuilder.population_size, 2, [(1, 20)]
    )
    (
        evoBuilder.set_mutation(MUTATION)
        .set_crossover(CROSSOVER)
        .set_selection(TournamentSelection(100, False))
    )
    evoBuilder.set_max_epoch(20)
    ff = FitnessFunction(
        sample_fitness_function, [(-10, 10)], 10, SimpleClampStrategy()
    )
    evoBuilder.set_fitness_function(ff)
    evoBuilder.set_representation(Representation(2))
    evoBuilder.create_evolution().run()


if __name__ == "__main__":
    main()
