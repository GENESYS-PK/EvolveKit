from core.Condition import Condition


class EpochCondition(Condition):
    def __init__(self, max_epoch: int):
        super().__init__()
        self.max_epoch = max_epoch

    def evaluate(self, evolution) -> bool:
        return evolution >= self.max_epoch