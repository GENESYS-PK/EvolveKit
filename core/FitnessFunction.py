import numpy as np
from collections.abc import Callable
from ClampStrategy import ClampStrategy
from Population import Population
from Individual import Individual

class FitnessFunction:
    """
    Class used to instantiate FitnessFunction object - responsible for 
    evaluating individuals of a given population.

    :param fitness_function: A function accepting NumPy array of floats that 
    returns a single float.
    :param variable_domains: A list of tuples of two floats, specifying upper 
    and lower bounds of values in chromosome.
    :param n_dim: Dimension (number of values) of a chromosome.
    :param clamp_strategy: An object responsible for enforcing domain 
    restrictions on a chromosome; inherits from ClampStrategy abstract class.
    """
    def __init__(
            self, 
            fitness_function: Callable[[np.ndarray[float]], float], 
            variable_domains: List[Tuple[float, float]], 
            n_dim: int, 
            clamp_strategy: ClampStrategy=Default):
        self.fitness_function = fitness_function
        self.variable_domains = variable_domains
        self.n_dim = n_dim
        self.clamp_strategy = clamp_strategy

    def calculate_individual_value(self, individual: Individual) -> float:
        """
        Method for a single individual's evaluation.

        :param individual: Instance of Individual class for evaluation.
        :return: _**float**_, an evaluation of individual. This method 
        does **not** change the _value_ field of individual.
        """
        return self.fitness_function(individual.chromosome)

    def eval_population(self, population: Population) -> None:
        """
        Evaluate entire population.

        :param population: Instance of Population class for evaluation.
        :return: _**None**_, evaluations are stored in each individual, 
        in _value_ field.
        """
        for i in population.population:
            i.value = self.fitness_function(i.chromosome)

    def get_eval_list(self, population: Population) -> np.ndarray[float]:
        """
        Evaluate entire population.

        :param population: Instance of Population class for evaluation.
        :return: A NumPy array of ***float***s, result of evaluation of 
        every individual in population. This method does **not** 
        change the _value_ field in any individual. 
        """
        ret = np.arange(population.population_size)
        for i in enumerate(population.population):
            ret[i[0]] = self.fitness_function(i[1].chromosome)
        return ret
    
    def clamp_population_to_domain(self, population: Population) -> None:
        """
        Force every individual in population into domain given by *variable_domains*.

        :param population: Instance of Population class to be clamped.
        """
        self.clamp_strategy(self.variable_domains, population)
