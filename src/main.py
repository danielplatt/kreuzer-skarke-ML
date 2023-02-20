import argparse

from src.models import *
from src.dataset import *
from config.constants import *

# --- Helper function ---

def create_model(
        args: dict,
        dataset: any
):
    """
    Wrapper function to create specified model.

    :param args: dict containing user specified inputs
    :type args: dict
    :param dataset: KreuzerSkarkeDataset object containing the type of dataset used. To directly use training and testing
     dataset, use X_train, Y_train, X_test, Y_test = dataset.X_train, dataset.Y_train, dataset.X_test, dataset.Y_test
    :type dataset: pytorch dataset object
    """
    model = args['model'].lower().strip()
    if model == 'invariantmlp':
        return InvariantMLP(dataset, output_tag=args['output_tag'], load_saved_model=args['eval'])
    elif model == 'hartford':
        return Hartford(dataset, output_tag=args['output_tag'], load_saved_model=args['eval'])
    elif model == 'xgboost':
        return XGboost(dataset,  output_tag=args['output_tag'], load_saved_model=args['eval'])
    else:
        raise ValueError('Unsupported model type %s' % model)


# --- Fetch user arguments ---

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataset', type=str, help='Dataset type. Should be from this list: ' + ', '.join(ALL_DATASETS))
parser.add_argument('model', type=str, help='Model Type. Should be from this list: ' + ', '.join(ALL_MODELS))
parser.add_argument('--output_tag', default=None, type=str, help='Output tag used to save results or fetch saved results. Necessary if "--eval" flag is used')
parser.add_argument('--eval', action='store_true', help='If specified, returns validation accuracy of saved model under "--output_tag"')
parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs for training')
args = vars(parser.parse_args())

# --- Create dataset and model objects ---

dataset = args['dataset'].lower().strip()
assert dataset in ALL_DATASETS

dataset = KreuzerSkarkeDataset(load_projections=True, projections_file=dataset)
model = create_model(args, dataset)

# --- Trigger training/evaluation ---

if not args['eval']:
    model.train(num_epochs=args['num_epochs'])
else:
    model.get_accuracy()