from EvolutionBuilder import EvolutionBuilder
from Evolution import Evolution
from Mutations.SphereMutation import SphereMutation
from Crossovers.OnePointAverageCrossover import OnePointAverageCrossover
from Selections.TournamentSelection import TournamentSelection
from FitnessFunction import FitnessFunction
from Clamps.SimpleClampStrategy import SimpleClampStrategy
import numpy as np
from Representation import Representation
from Population import Population
from Individual import Individual
from typing import List, Tuple

def sample_fitness_function(chromosome: np.ndarray) -> float:
    return np.sum(chromosome)

def custom_population_generator(
        population_size: int,
        individual_size: int,
        variable_domains: List[Tuple[int, int]]
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
    evoBuilder.set_population_size(100).set_maximize(False)
    evoBuilder.set_population_generator(custom_population_generator(100, 10, [(1, 10)]), 100, 10, (1, 10))
    evoBuilder.set_mutation(SphereMutation(0.1)).set_crossover(OnePointAverageCrossover(2, 0.9)).set_selection(TournamentSelection(100, False))
    ff = FitnessFunction(sample_fitness_function, [(-10, 10)], 10, SimpleClampStrategy())
    evoBuilder.set_fitness_function(ff)
    evoBuilder.set_representation(Representation(2))
    evoBuilder.create_evolution().run()


if __name__ == '__main__':
    main()