import os
from pathlib import Path
from typing import List


class PathSearcher:

    @staticmethod
    def get_list_of_dataset_paths(path: Path) -> List[Path]:
        paths = list(path.rglob('*.csv'))
        return paths
    
    @staticmethod
    def get_path_of_datasets_with_timestamp_if_possible(path: Path, timestamp: str = None) -> List[Path]:
        path_to_timestamp = path / timestamp
        if not os.path.exists(path_to_timestamp):
            raise FileNotFoundError(path_to_timestamp)
        return path_to_timestamp