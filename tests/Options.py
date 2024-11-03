from core.impl.operators.crossovers import BlendCrossoverAlfaBeta
from core.impl.operators.mutations import ModifiedUniformMutation, DebGoyalMutation

MUTATION = DebGoyalMutation(3, 0.6)
CROSSOVER = BlendCrossoverAlfaBeta(10, 0.5)
