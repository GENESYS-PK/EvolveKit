from collections.abc import Callable
from core.evolution import Evolution
from core.expression import Expression

class Job(Expression):
    """
    Represents a logical statement followed by an execution of user defined
    callback. Used to modify main evolution sequence when certain criteria are
    met (time elapsed, epoch reached, etc.).
    """

    def __init__(self, action: Callable[[Evolution], None]):
        """
        Instantiate job.
        :param action: A function callback to execute when expression criteria are met.
        """
        super().__init__()
        self.action = action

    def handle(self, evolution: Evolution) -> None:
        """
        Check if job callback should be executed. If so, do as such.
        :param evolution: An Evolution instance to check against.
        """
        if self.evaluate(evolution):
            self._run_action(evolution)

    def _run_action(self, evolution: Evolution) -> None:
        """
        Internal method to execute callback.
        :param evolution: An Evolution instance on which to execute callback.
        """
        self.action(evolution)

    def set_callback(self, action: Callable[[Evolution], None]) -> None:
        """
        Set executed callback.
        :param action: A function callback to execute when expression criteria are met.
        """
        self.action = action