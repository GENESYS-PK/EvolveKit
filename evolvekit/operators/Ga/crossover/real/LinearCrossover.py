from typing import List
import numpy as np
import copy

from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum

class LinearCrossover(GaOperator):
    def __init__(self):
        """Initialize the LinearCrossover operator.""" 
        super().__init__()
    
    def category(self) -> GaOpCategory:
        """Returns the category of the operator.

        :returns: GaOpCategory indicating real crossover operator type
        """
        return GaOpCategory.REAL_CROSSOVER

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """Performs linear crossover on two randomly selected parents
        from the population.

        Creates three children using different linear combinations:
        - child_Z = 0.5 * parent_1 + 0.5 * parent_2
        - child_V = 1.5 * parent_1 - 0.5 * parent_2
        - child_W = -0.5 * parent_1 + 1.5 * parent_2

        :param args: GaOperatorArgs containing population and evaluator
        :returns: List of two best GaIndividual offspring based on fitness evaluation
        :raises IndexError: when population has fewer than 2 individuals for parent selection
        """
        child_Z = GaIndividual()
        child_V = GaIndividual()
        child_W = GaIndividual()
        parent_1, parent_2 = np.random.choice(args.population, size=2, replace=False)
        child_Z.real_chrom = 0.5 * parent_1.real_chrom + 0.5 * parent_2.real_chrom
        child_V.real_chrom = 1.5 * parent_1.real_chrom - 0.5 * parent_2.real_chrom
        child_W.real_chrom = -0.5 * parent_1.real_chrom + 1.5 * parent_2.real_chrom
        child_Z.value = args.evaluator.evaluate(GaEvaluatorArgs(child_Z))
        child_V.value = args.evaluator.evaluate(GaEvaluatorArgs(child_V))
        child_W.value = args.evaluator.evaluate(GaEvaluatorArgs(child_W))
        pottential_children = [child_Z, child_V, child_W]
        pottential_children.sort(key=lambda ind: ind.value, reverse=args.evaluator.extremum() == GaExtremum.MINIMUM)
        return pottential_children[:2]
