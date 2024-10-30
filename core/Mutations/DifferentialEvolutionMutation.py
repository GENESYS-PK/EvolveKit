import numpy as np
from core.Mutation import Mutation
from core.Population import Population


class DifferentialEvolutionMutation(Mutation):
    def __init__(self, fitness_function: callable, minimize: bool = True, probability: float = 0):
        """
        Constructor for the Differential Evolution Mutation (DEM) class.

        :param probability: Probability of mutation (float between 0 and 1).
        :param fitness_function: A callable fitness function that evaluates individuals.
        :param minimize: Whether the mutation is for a minimization problem (default True)
                         or maximization (False).
        """
        super().__init__(probability)
        self.fitness_function = fitness_function
        self.minimize = minimize

    def mutate(self, population_parent: Population) -> Population:
        """
        Perform the Differential Evolution Mutation operations for the entire population.

        :param population_parent: The population to perform the mutation operation on.
        :returns: The mutated population.
        """
        self._mutate(population_parent)

        return population_parent

    def _mutate(self, population: Population) -> None:
        """
        Perform the Differential Evolution Mutation operations for the entire population.

        :param population: The population containing individuals.
        :returns: None.
        """
        population_size = len(population.population)
        fitness = []

        # Find the best individual based on minimization or maximization
        best_fitness = float('inf') if self.minimize else -float('inf')
        best_index = 0
        for i in range(population_size):
            ind_fitness = self.fitness_function(population.population[i].chromosome)
            fitness.append(ind_fitness)
            if (self.minimize and ind_fitness < best_fitness) or (not self.minimize and ind_fitness > best_fitness):
                best_fitness = ind_fitness
                best_index = i

        # Select 4 random vectors from population (excluding the best one)
        vectors = []
        while len(vectors) < 4:
            random_index = np.random.randint(0, population_size)
            if random_index != best_index:
                vectors.append(population.population[random_index].chromosome)

        # Differential vector
        rz_vector = (vectors[0] - vectors[1]) + (vectors[2] - vectors[3])
        alpha = np.random.uniform(0, 1.2)  # Scale factor

        # Create the perturbed best individual
        best_perturbed_individual = population.population[best_index].chromosome + alpha * rz_vector

        # Select one random individual Y from the population
        index_Y = np.random.randint(0, population_size)
        ind_Y = population.population[index_Y].chromosome
        size_Y = len(ind_Y)

        # Create a new individual Y' based on the perturbed best individual
        new_ind_Y = np.copy(ind_Y)
        for i in range(size_Y - 1):
            if np.random.uniform(0, 1) < self.probability:
                new_ind_Y[i] = best_perturbed_individual[i]

        # Ensure the last gene is always copied
        new_ind_Y[size_Y - 1] = best_perturbed_individual[size_Y - 1]

        # Replace the individual if the new one has better fitness
        new_fitness_Y = self.fitness_function(new_ind_Y)
        if (self.minimize and new_fitness_Y <= fitness[index_Y]) or (
                not self.minimize and new_fitness_Y >= fitness[index_Y]):
            population.population[index_Y].chromosome = new_ind_Y