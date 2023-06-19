import os, sys, json, numpy as np, pandas as pd, pickle, itertools, logging
import seaborn as sns, matplotlib.pyplot as plt, matplotlib.patches as mpatches

def make_hodge_patches(categorical_values, cmap):
    cats = sorted(list(np.unique(categorical_values)))
    n_cats = len(cats)
    return [mpatches.Patch(color=cmap(i  /(n_cats - 1)), label=cat) for i, cat in enumerate(cats)]

def tripple_scatter(df:pd.DataFrame, emb_name:str='PHATE', figsize=(14, 8), dpi:int=300, cmap:str='viridis'):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1,1,1)
    
    ax.set_title(f'{emb_name} Embedding of Hodge Numbers')
    ax.set_xlabel(f'{emb_name} 1')
    ax.set_ylabel(f'{emb_name} 2')

    ax.scatter(df.d1, df.d2, s=120, c='black', cmap=cmap)
    ax.scatter(df.d1, df.d2, s=100, c='white', cmap=cmap)
    ax.scatter(df.d1, df.d2,  s=80, c=df.Hodge, cmap=cmap, alpha=0.7, marker='o', edgecolors=None)
    
    cmap = plt.get_cmap()
    fig.legend(handles=make_hodge_patches(df.Hodge, cmap))
    return fig, ax