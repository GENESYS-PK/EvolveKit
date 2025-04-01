import numpy as np
from core.evolution import EvolutionBuilder
from core.evolution import Evolution
from core.fitness_function import FitnessFunction
from core.impl.clamps import SimpleClampStrategy
from core import Representation
from core import Population
from core import Individual
from typing import List, Tuple
from core.impl.operators.selections import *
from core.impl.operators.crossovers import *
from core.impl.operators.mutations import *

# -------------------------- CONFIG --------------------------
def Rosenbrock(x: np.ndarray) -> float:
    return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))

# Minimum for Rosenbrock function is 0 at all arguments equal to 1

population_size = 40
select_size = 20
generations = 20
domain = np.array([(-5, 5), (-5, 5), (-5, 5)])
maximize = False

clamp_strategy = SimpleClampStrategy()
ff = FitnessFunction(Rosenbrock, domain, len(domain), clamp_strategy)

selection = TournamentSelection(select_size, maximize, 3)
crossover = LinearCrossover(population_size, ff, 0.9, maximize)
mutation = UniformMutation(np.array([-5, 5]), 0.2) # TODO: pass "domain"
# ------------------------------------------------------------

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

        for i in range(individual_size):
            domain = variable_domains[
                i % len(variable_domains)
            ]  # Repeat the domains if needed
            min_value, max_value = domain

            gene_value = np.random.uniform(min_value, max_value)
            individual_genes.append(gene_value)

        individual = Individual(individual_genes, gene_value)
        individuals.append(individual)

    return Population(population=individuals)

def main():
    evoBuilder = EvolutionBuilder(Evolution)
    evoBuilder.set_population_size(population_size).set_maximize(maximize)
    evoBuilder.set_selection(selection)
    evoBuilder.set_crossover(crossover)
    evoBuilder.set_mutation(mutation)
    evoBuilder.set_max_epoch(generations)
    evoBuilder.set_fitness_function(ff)
    evoBuilder.set_population_generator(custom_population_generator, population_size, len(domain), domain)
    evoBuilder.set_representation(Representation.REAL)
    evolution = evoBuilder.create_evolution()
    evolution.run()

if __name__ == "__main__":
    main()
