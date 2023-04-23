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
    
    sorted_indices = np.argsort(counts_truth)[::-1]
    sorted_cl = unique_truth[sorted_indices]
    median_index = len(counts_truth)//2
    tail, bulk = sorted_cl[:median_index], sorted_cl[median_index:]
    
    truth_pre = []
    preds_pre = []
    for i in range(len(truth)):
        if truth[i] in tail:
            truth_pre.append(0)
        else:
            truth_pre.append(1)
        
        if preds[i] in tail:
            preds_pre.append(0)
        else:
            preds_pre.append(1)
    
    plt.figure(figsize=(25,20))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm, annot=True, fmt='.1%', cmap="viridis", 
                xticklabels=classes, yticklabels=classes, cbar=False).invert_yaxis()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("_full.".join(str(confm_path).split(".")))
    plt.close()
    
    cm = confm(truth_pre, preds_pre)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm, annot=True, fmt='.1%', cmap="viridis", 
                xticklabels=["tail", "bulk"], yticklabels=["tail", "bulk"], cbar=False).invert_yaxis()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(confm_path)
    
    return