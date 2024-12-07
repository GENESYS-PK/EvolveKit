from core.expression.JobManager import JobManager
from core.operators.Mutation import Mutation
from core.operators.Selection import Selection
from core.operators.Crossover import Crossover
from core.operators.Elitism import Elitism
from core.fitness_function.FitnessFunction import FitnessFunction
from core.expression.Job import Job
from core.Population import Population
from core.expression.Expression import Expression
from core.evolution.EvolutionState import EvolutionState
from core.operators.OperatorsPreset import OperatorsPreset


from typing import List, Tuple, Callable, Self


class Evolution:
    def __init__(
        self,
        mutation: Mutation = None,
        selection: Selection = None,
        crossover: Crossover = None,
        elitism: Elitism = None,
        fitness_function: FitnessFunction = None,
        job_queue: JobManager = None,
        init_population: Population = None,
        population_size: int = None,
        terminator: Expression = None,
        maximize: bool = None,
        events: Tuple[List[Callable[[EvolutionState], None]], ...] = None,
    ):
        self.mutation = mutation
        self.selection = selection
        self.crossover = crossover
        self.elitism = elitism
        self.fitness_function = fitness_function
        self.job_queue = job_queue if job_queue is not None else JobManager()
        self.terminator = terminator
        self.evolution_state = EvolutionState([], [], [], False, 0)
        self.terminate_loop: bool = False
        self.events = events
        self.representation = []
        self.init_population = init_population
        self.population_size = population_size
        self.maximize = maximize

    def set_selection(self, selection: Selection) -> Self:
        self.selection = selection
        return self

    def set_crossover(self, crossover: Crossover) -> Self:
        self.crossover = crossover
        return self

    def set_mutation(self, mutation: Mutation) -> Self:
        self.mutation = mutation
        return self

    def use_preset(self, preset: OperatorsPreset) -> Self:
        self.selection = preset.selection
        self.crossover = preset.crossover
        self.mutation = preset.mutation
        return self

    def run(self) -> None:
        self.prepare_evolution_state()
        while not self.terminate_loop:
            self.loop()

            print(f" --- Generation {self.evolution_state.current_epoch} ---")
            for indiv in self.evolution_state.current_population.population:
                print(f"{indiv.chromosome} -> {indiv.value}")
            print()

            if self.terminator.evaluate(self.evolution_state.current_epoch):
                break

    def loop(self) -> None:
        self.fitness_function.eval_population(self.evolution_state.current_population)
        self.step_selection()
        self.step_crossover()
        self.fitness_function.eval_population(self.evolution_state.current_population)
        self.step_mutation()
        self.fitness_function.eval_population(self.evolution_state.current_population)
        self.evolution_state.update_evolution_state()
        self.job_queue.evaluate_jobs(self)

    def get_evolution_state(self) -> EvolutionState:
        return self.evolution_state

    def prepare_evolution_state(self) -> None:
        self.evolution_state.current_population = self.init_population
        self.evolution_state.maximize = self.maximize
        self.evolution_state.population_size = self.population_size

    def perform_crossover(self) -> None:
        self.evolution_state.current_population = self.crossover.cross(
            self.evolution_state.current_population
        )

    def perform_selection(self) -> None:
        self.evolution_state.current_population = self.selection.select(
            self.evolution_state.current_population
        )

    def perform_mutation(self) -> None:
        self.evolution_state.current_population = self.mutation.mutate(
            self.evolution_state.current_population
        )

    def step_selection(self) -> None:
        self.perform_selection()

    def step_crossover(self) -> None:
        self.perform_crossover()

    def step_mutation(self) -> None:
        self.perform_mutation()
