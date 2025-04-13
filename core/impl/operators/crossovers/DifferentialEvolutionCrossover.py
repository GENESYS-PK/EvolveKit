import numpy as np

from core import Population, Individual
from core.operators import Crossover
from core.Population import Population
from core.fitness_function import FitnessFunction
from core.Representation import Representation


class DifferentialEvolutionCrossover(Crossover):
    """
    Implements Differential Evolution Crossover based on the algorithm described.

    :param how_many_individuals: The number of individuals to create in the offspring.
    :param fitness_function: A function that is being optimized.
    :param crossover_rate: The probability of performing the crossover operation.
    :param minimize: If True, the algorithm minimizes the fitness function; if False, it maximizes.
    """

    allowed_representation = [Representation.REAL]

    def __init__(
        self,
        how_many_individuals: int,
        fitness_function: FitnessFunction,
        crossover_rate: float = 0.7,
        minimize: bool = True,
    ):
        super().__init__(how_many_individuals, crossover_rate)
        self.fitness_function = fitness_function
        self.crossover_rate = crossover_rate
        self.minimize = minimize 

    def _cross(self, population_parent: Population) -> Population:
        """
        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring population.
        """
        new_population = []
        pop_size = len(population_parent.population)
        dim = len(population_parent.population[0].chromosome)

        if self.minimize:
            best_individual = min(
                population_parent.population,
                key=lambda ind: self.fitness_function.calculate_individual_value(ind),
            )
        else:
            best_individual = max(
                population_parent.population,
                key=lambda ind: self.fitness_function.calculate_individual_value(ind),
            )

        best_chromosome = np.array(best_individual.chromosome)

        for i in range(pop_size):
            indices = list(range(pop_size))
            indices.remove(i)
            a, b, c, d = np.random.choice(indices, 4, replace=False)

            chrom_a = np.array(population_parent.population[a].chromosome)
            chrom_b = np.array(population_parent.population[b].chromosome)
            chrom_c = np.array(population_parent.population[c].chromosome)
            chrom_d = np.array(population_parent.population[d].chromosome)

            Rz = (chrom_a - chrom_b) + (chrom_c - chrom_d)

            scaling_factor = np.random.uniform(0, 1.2)
            perturbed_best = best_chromosome + scaling_factor * Rz

            offspring_chromosome = np.array(population_parent.population[i].chromosome)

            for j in range(dim - 1):
                if np.random.uniform(0, 1) < self.crossover_rate:
                    offspring_chromosome[j] = perturbed_best[j]

            offspring_chromosome[-1] = perturbed_best[-1]

            offspring = Individual(chromosome=offspring_chromosome.tolist())

            parent_fitness = self.fitness_function.calculate_individual_value(population_parent.population[i])
            offspring_fitness = self.fitness_function.calculate_individual_value(offspring)

            if self.minimize:
                if offspring_fitness <= parent_fitness:
                    new_population.append(offspring)
                else:
                    new_population.append(population_parent.population[i])
            else:
                if offspring_fitness >= parent_fitness:
                    new_population.append(offspring)
                else:
                    new_population.append(population_parent.population[i])

        return Population(new_population)
