from pathlib import Path

BASE_DIR = Path(__file__).parents[1]

ZIPPED_INPUT_FILE = BASE_DIR.joinpath('data/raw/v26.gz')
UNZIPPED_INPUT_FILE = BASE_DIR.joinpath('data/raw/v26')

PROJECTION_DIR = BASE_DIR.joinpath('data/projections')
SAVED_MODELS_DIR = BASE_DIR.joinpath('data/saved_models')
SAVED_RESULTS_DIR = BASE_DIR.joinpath('data/saved_results')
TENSORBOARD_DIR = BASE_DIR.joinpath('data/runs/tensorboard')

ALL_DATASETS = [
    'original',
    'original_permuted',
    'combinatorial',
    'combinatorial_permuted',
    'dirichlet',
    'dirichlet_permuted',
]

ALL_MODELS = [
    'invariantmlp',
    'hartford',
]