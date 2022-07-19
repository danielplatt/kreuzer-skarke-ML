from pathlib import Path

BASE_DIR = Path(__file__).parents[1]

ZIPPED_INPUT_FILE = BASE_DIR.joinpath('data/raw/v26.gz')
UNZIPPED_INPUT_FILE = BASE_DIR.joinpath('data/raw/v26')

PROJECTION_DIR = BASE_DIR.joinpath('data/projections')