import os
from pathlib import Path
from typing import List



class PathSearcher:

    @staticmethod
    def get_list_of_dataset_paths(path: Path, filetype) -> List[Path]:
        if not path.exists():
            print("Path does not exist:  " + str(path))
        paths = list(path.rglob(filetype))
        return paths
    
    @staticmethod
    def get_path_of_datasets_with_timestamp_if_possible(path: Path, timestamp: str = None) -> List[Path]:
        path_to_timestamp = path / timestamp
        if not os.path.exists(path_to_timestamp):
            raise FileNotFoundError(path_to_timestamp)
        return path_to_timestamp
    

    @staticmethod
    def get_list_of_subdirectories(path: Path):
        directory_contents = os.listdir(path)
        return directory_contents