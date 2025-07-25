from evolvekit.core.Ga.enums import GaClampStrategy
from evolvekit.core.Ga.GaEvaluator import GaEvaluator
from evolvekit.core.Ga.GaInspector import GaInspector
from evolvekit.core.Ga.GaResults import GaResults
from evolvekit.core.Ga.GaState import GaState
from evolvekit.core.Ga.operators.GaOperator import GaOperator


class GaIsland(GaState):
    """
    Class representing a single, self contained simulation environment.
    """

    inspector: GaInspector
    selection: GaOperator
    real_crossover: GaOperator
    real_mutation: GaOperator
    bin_crossover: GaOperator
    bin_mutation: GaOperator

    def __verify(self):
        """
        Self-verifies integrity of provided data.

        Checks whether all required data was provided. If not, an
        appropriate exception will be thrown.

        :throws: TODO list all possible exceptions
        """

        pass

    def __initialize(self):
        """
        Initializes evolution.

        TODO describe in more detail what is being initialized here.
        """

        pass

    def __evaluate(self):
        """
        Runs fitness evaluation on every individual.
        """

        pass

    def __evolve(self):
        """
        Generates next population.

        TODO describe exactly what is being done here.
        """

        pass

    def __finish(self) -> GaResults:
        """
        Finishes current simulation and returns its results.

        :returns: Object representing final result of running genetic algorithm.
        :rtype: :class:`GaResults`.
        """

        pass

    def run(self) -> GaResults:
        """
        Run entire simulation.

        :returns: Object representing final result of running genetic algorithm.
        :rtype: :class:`GaResults`.
        """

        pass

    def set_elite_count(self, count: int):
        """
        Setter method.

        Set the number of elite individuals passed onto the next generation.

        :param count: Number of elite individuals.
        :type count: int.
        """

        pass

    def set_crossover_probability(self, prob: float):
        """
        Setter method.

        Set the probability of crossover.

        :param prob: The probability [0.0, 1.0].
        :type prob: float.
        """

        pass

    def set_mutation_probability(self, prob: float):
        """
        Setter method.

        Set the probability of mutation.

        :param prob: The probability [0.0, 1.0].
        :type prob: float.
        """

        pass

    def set_max_generations(self, count: int):
        """
        Setter method.

        Set the maximum number of generations.

        :param count: Generation number after which simulation should end.
        :type count: int.
        """

        pass

    def set_seed(self, seed: int):
        """
        Setter method.

        Set the simulation seed.

        :param seed: A provided seed.
        :type seed: int.
        """

        pass

    def set_evaluator(self, evaluator: GaEvaluator):
        """
        Setter method.

        Set the evaluator object used by the simulation.

        :param evaluator: Object representing valid evaluator.
        :type evaluator: :class:`GaEvaluator`.
        """

        pass

    def set_inspector(self, inspector: GaInspector):
        """
        Setter method.

        Set the inspector object used by the simulation.

        :param inspector: Object representing valid inspector.
        :type inspector: :class:`GaInspector`.
        """

        pass

    def set_operator(self, operator: GaOperator):
        """
        Setter method.

        Set the operator acting on the population.

        :param operator: An operator used by genetic algorithm.
        :type operator: :class:`GaOperator`.
        """

        pass

    def set_real_clamp_strategy(self, strategy: GaClampStrategy):
        """
        Setter method.

        Set the clamping strategy used by the current simulation.

        :param strategy: A value representing chosen clamping strategy.
        :type strategy: :class:`GaClampStrategy`.
        """

        pass

    def set_population_size(self, size: int):
        """
        Setter method.

        Set the maximum number of individuals in this simulation.

        :param size: Number of individuals.
        :type size: int. 
        """

        pass
