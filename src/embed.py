import argparse

import os, sys, json, numpy as np, pandas as pd
from sklearn.decomposition import PCA
import phate


from config.constants import UNZIPPED_INPUT_FILE
from src.dataset import KreuzerSkarkeDataset
from utils.plots import tripple_scatter


from src.dataset import *
from config.constants import *
from utils.projections import *

from config.constants import *


# --- Fetch user arguments ---

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataset', type=str, help='Dataset type. Should be from this list: ' + ', '.join(ALL_DATASETS))
parser.add_argument('--num_comps', default=10, type=int, help='Number of components to embed too')
parser.add_argument('--save', action='store_true', help='If specified, saves results')
args = vars(parser.parse_args())


# --- Create dataset and model objects ---

dataset = args['dataset'].lower().strip()
assert dataset in ALL_DATASETS

dataset = KreuzerSkarkeDataset(load_projections=True, projections_file=dataset)

X_proj, Y = dataset.X_proj, dataset.Y

n_comps = int(args['num_comps'])
c_names = [f'd{i}' for i in range(1, n_comps + 1)]

X_flat = X_proj.reshape(-1, np.multiply(*list(X_proj.shape)[1:]))

phate_op = phate.PHATE(n_components=n_comps, n_jobs=-1)
Y_phate = phate_op.fit_transform(X_flat)

pca_op = PCA(n_components=n_comps)
Y_pca = pca_op.fit_transform(X_flat)

df_phate = pd.DataFrame(Y_phate, columns=c_names).join(pd.DataFrame(Y, columns=['Hodge']))
df_pca  = pd.DataFrame(Y_pca, columns=c_names).join(pd.DataFrame(Y, columns=['Hodge']))



if args['save']:
    rawpath = PROJECTION_DIR.joinpath('df_phate.pkl')
    print('Saving PHATE DataFrame to % s' % rawpath)
    df_phate.to_pickle(rawpath)

    
    rawpath = PROJECTION_DIR.joinpath('df_pca.pkl')
    print('Saving PCA DataFrame to % s' % rawpath)
    df_pca.to_pickle(rawpath)
    

    rawpath = PROJECTION_DIR.joinpath('phate.png')    
    print('Saving PHATE Embedding to % s' % rawpath)
    fig, ax = tripple_scatter(df_phate, emb_name='PHATE')
    fig.savefig(rawpath)

    
    rawpath = PROJECTION_DIR.joinpath('pca.png')    
    print('Saving PCA Embedding to % s' % rawpath)
    fig, ax = tripple_scatter(df_pca, emb_name='PCA')
    fig.savefig(rawpath)
