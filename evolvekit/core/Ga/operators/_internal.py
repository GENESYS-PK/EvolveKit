from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory

CATEGORY_TO_POPULATION_FIELD = {
    GaOpCategory.SELECTION: "current_population",
    GaOpCategory.REAL_CROSSOVER: "selected_population",
    GaOpCategory.REAL_MUTATION: "offspring_population",
    GaOpCategory.BIN_CROSSOVER: "selected_population",
    GaOpCategory.BIN_MUTATION: "offspring_population",
}
"""
Should be the same as member names in `GaState` class.
"""