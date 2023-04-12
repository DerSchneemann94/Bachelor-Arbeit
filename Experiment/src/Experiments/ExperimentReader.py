from pathlib import Path
from utils import get_project_root
import yaml


class ExperimentReader:

    def __init__(self, path: Path) -> None:
        self._experiment_config = self.__create_experiment_from_config_file(path)

   
    def __create_experiment_from_config_file(self, path: Path):
        try:
            with open(path, "r") as file: 
                experiment_config = yaml.load(file, yaml.Loader)
                return experiment_config
        except Exception as error:
            print("Experiment-Confguration at " + str(path) + " could not be loaded.")
            raise error


    def get_experiment_config(self):
        return self._experiment_config


if __name__ == "__main__":
    path = get_project_root() / "src/Experiments/ExperimentFiles/experiment_openml_jeager_by_hand.yaml"
    ExperimentReader.create_experiment_from_config_file(path)