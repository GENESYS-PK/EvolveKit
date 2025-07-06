from evolvekit.core.Ga.enums import GaClampStrategy
from evolvekit.core.Ga.GaEvaluator import GaEvaluator
from evolvekit.core.Ga.GaInspector import GaInspector
from evolvekit.core.Ga.GaResults import GaResults
from evolvekit.core.Ga.GaState import GaState
from evolvekit.core.Ga.operators.GaOperator import GaOperator


class GaIsland(GaState):
    inspector: GaInspector
    selection: GaOperator
    real_crossover: GaOperator
    real_mutation: GaOperator
    bin_crossover: GaOperator
    bin_mutation: GaOperator

    def __verify(self):
        pass

    def __initialize(self):
        pass

    def __evaluate(self):
        pass

    def __evolve(self):
        pass

    def __finish(self) -> GaResults:
        pass

    def run(self) -> GaResults:
        pass

    def set_elite_count(self, count: int):
        pass

    def set_crossover_probability(self, prob: float):
        pass

    def set_mutation_probability(self, prob: float):
        pass

    def set_max_generations(self, count: int):
        pass

    def set_seed(self, seed: int):
        pass

    def set_evaluator(self, evaluator: GaEvaluator):
        pass

    def set_inspector(self, inspector: GaInspector):
        pass

    def set_operator(self, operator: GaOperator):
        pass

    def set_real_clamp_strategy(self, strategy: GaClampStrategy):
        pass

    def set_population_size(self, size: int):
        pass
