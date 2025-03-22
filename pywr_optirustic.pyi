from pathlib import Path

from optirustic import Algorithm, ExportHistory

class OptimisationScenario:
    """
    Class to define an optimisation scenarios to set problem objectives and
    constraints.
    """

    @classmethod
    def load_from_file(cls, path: Path) -> OptimisationScenario:
        """
        Load the optimisation scenario from a JSON file.
        :param path: The path to the JSON file.
        :return: The class instance.
        """
        ...

class PywrOptirustic:
    """Class used to initialise and run an optimisation scenario."""

    @staticmethod
    def run(
        model_file: Path,
        algorithm: Algorithm,
        export_history: ExportHistory,
        scenario_file: Path,
        data_path: Path | None = None,
    ) -> None:
        """
        Solve the problem with the chosen algorithm and pywr model.
        :param model_file: The path to the model JSON file.
        :param algorithm: The algorithm and its option to use to solve the optimisation
        problem.
        :param export_history: The data used to configure the result export for the
        generations. The option in 'algorithm.export_history', if provided, is
        overwritten by this argument.
        :param scenario_file: The optimisation scenario file to use to define the
        objectives and constraints.
        :param data_path: The optional path where the model data is stored.
        :return: None.
        """
