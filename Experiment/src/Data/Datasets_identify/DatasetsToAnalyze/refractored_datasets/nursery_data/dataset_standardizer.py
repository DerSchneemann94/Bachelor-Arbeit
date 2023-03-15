import json
from pathlib import Path
from utils import get_project_root

datasets_path = get_project_root() / "src/Data/Datasets_identify/DatasetsToAnalyze"
source_path = datasets_path / "original_data_sources/nursery_data/nursery.txt"
destination_path = datasets_path / "refractored_datasets/nursery_data/nursery.txt"
data = source_path.read_text()
data = data.replace(' ', '')
destination_path.write_text(data)