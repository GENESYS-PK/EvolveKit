import csv
from typing import Optional

from evolvekit.core.Ga.GaInspector import GaInspector
from evolvekit.core.Ga.GaStatistics import GaStatistics
from evolvekit.core.Ga.enums.GaAction import GaAction


class CSVInspector(GaInspector):
    """
    Inspector that logs generation statistics to a CSV file.
    """

    def __init__(
            self,
            filename: str,
            stagnation_limit: int = 100,
            generation_offset: int = 1,
    ):
        """
        Initialize the CSV inspector.

        :param filename: output CSV file path
        :param stagnation_limit: max generations without improvement
        :param generation_offset: value subtracted from `stats.generation`
            when logging, so you can start from 1.
        """
        self.filename: str = filename
        self.stagnation_limit: int = stagnation_limit
        self.generation_offset: int = generation_offset
        self._csv_file: Optional[object] = None
        self._csv_writer: Optional[csv._writer] = None
        self._previous_best: Optional[float] = None

    def initialize(self):
        """
        Open the CSV file and write the header row.
        """
        self._csv_file = open(self.filename, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            [
                "Generation",
                "Best_Fitness",
                "Worst_Fitness",
                "Mean",
                "Median",
                "StdDev",
                "Stagnation",
                "Time_Elapsed",
                "Improvement",
            ]
        )

    def inspect(self, stats: GaStatistics) -> GaAction:
        """
        Log current generation statistics and decide whether to continue.

        :param stats: current generation statistics
        :returns: GaAction.CONTINUE or GaAction.TERMINATE
        """
        improvement = 0.0
        if self._previous_best is not None and stats.best_indiv:
            improvement = self._previous_best - stats.best_indiv.value

        gen_to_log = max(1, stats.generation - self.generation_offset)

        self._csv_writer.writerow(
            [
                gen_to_log,
                stats.best_indiv.value if stats.best_indiv else 0.0,
                stats.worst_indiv.value if stats.worst_indiv else 0.0,
                stats.mean,
                stats.median,
                stats.stdev,
                stats.stagnation,
                stats.last_time - stats.start_time,
                improvement,
            ]
        )
        self._csv_file.flush()

        if stats.best_indiv:
            self._previous_best = stats.best_indiv.value

        if stats.stagnation >= self.stagnation_limit:
            return GaAction.TERMINATE

        return GaAction.CONTINUE

    def finish(self, stats: GaStatistics):
        """
        Close the CSV file.
        """
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
