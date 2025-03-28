import numpy as np

from core import Population, Individual
from core.operators import Crossover
from core.fitness_function import FitnessFunction
from core.Representation import Representation


class DerivativeBasedCrossover(Crossover):
    """krzyżowanie z pochodną (Derivative-Based Crossover)."""
    
    allowed_representation = [Representation.REAL]
    
    def __init__(self, how_many_individuals: int, fitness_function: FitnessFunction, crossover_rate: float = 0.7, epsilon: float = 1e-6):
        super().__init__(how_many_individuals, crossover_rate)
        self.fitness_function = fitness_function
        self.crossover_rate = crossover_rate
        self.epsilon = epsilon

    def _approximate_gradient(self, chromosome: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(chromosome)
        base_value = self.fitness_function.calculate(chromosome)
        for i in range(len(chromosome)):
            perturbed = np.copy(chromosome)
            perturbed[i] += self.epsilon
            grad[i] = (self.fitness_function.calculate(perturbed) - base_value) / self.epsilon
        return grad

    def _cross(self, population_parent: Population) -> Population:
        new_population = []
        pop_size = len(population_parent.population)
        dim = len(population_parent.population[0].chromosome)
        
        for i in range(pop_size):
            indices = list(range(pop_size))
            indices.remove(i)
            partner_idx = np.random.choice(indices)
            parent = population_parent.population[i]
            partner = population_parent.population[partner_idx]
            
            grad = self._approximate_gradient(np.array(parent.chromosome))
            
            alpha = np.random.uniform(0, 1.0)
            beta = np.random.uniform(0, 1.2)
            
            candidate = np.array(parent.chromosome) - alpha * grad + beta * (np.array(partner.chromosome) - np.array(parent.chromosome))
            
            offspring_chromosome = np.copy(parent.chromosome)
            for j in range(dim - 1):
                if np.random.uniform(0, 1) < self.crossover_rate:
                    offspring_chromosome[j] = candidate[j]
                    
            offspring_chromosome[-1] = candidate[-1]
            
            offspring = Individual(chromosome=offspring_chromosome)
            
            # Selekcja: jeżeli potomek osiąga lepszy (niższy) wynik funkcji celu, zostaje wybrany
            if self.fitness_function.calculate_individual_value(offspring) <= self.fitness_function.calculate_individual_value(parent):
                new_population.append(offspring)
            else:
                new_population.append(parent)
                
        return Population(new_population)
