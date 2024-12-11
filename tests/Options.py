import numpy as np
from core.impl.clamps import SimpleClampStrategy
from core.impl.operators.crossovers import (
    BlendCrossoverAlfaBeta,
    AdaptiveProbabilityOfGeneCrossover,
)
from core.impl.operators.mutations import ModifiedUniformMutation, DebGoyalMutation
from core.fitness_function import FitnessFunction


MUTATION = DebGoyalMutation(3, 0.6)
CROSSOVER = BlendCrossoverAlfaBeta(10, 0.5)
