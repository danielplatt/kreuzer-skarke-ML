from sklearn.metrics import confusion_matrix as confm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def confusion_matrix(preds, truth, confm_path):
    cm = confm(truth, preds)
    
    unique_truth, counts_truth = np.unique(truth, return_counts=True)
    unique_preds = np.unique(preds, return_counts=False)
    unique = list(set(np.concatenate((unique_preds, unique_truth), axis=0)))
    classes = [str(x) for x in unique]
    
    plt.figure(figsize=(25,20))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm, annot=True, fmt='.1%', cmap="viridis", 
                xticklabels=classes, yticklabels=classes, cbar=False).invert_yaxis()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    plt.savefig(confm_path)
    return