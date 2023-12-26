# import the necessary packages
from pyimagesearch import config
import matplotlib.pyplot as plt
import sklearn.metrics
import numpy as np

def compute_precision_recall(yTrue, predScores, thresholds):
    precisions = []
    recalls = []

    # loop over each threshold from 0.2 to 0.65
    for threshold in thresholds:
        # yPred is dog if prediction score greater than threshold
        # else cat if prediction score less than threshold
        yPred = [
            "dog" if score >= threshold else "cat"
            for score in predScores
        ]
  
        # compute precision and recall for each threshold
        precision = sklearn.metrics.precision_score(y_true=yTrue,
            y_pred=yPred, pos_label="dog")
        recall = sklearn.metrics.recall_score(y_true=yTrue,
            y_pred=yPred, pos_label="dog")
  
        # append precision and recall for each threshold to
        # precisions and recalls list
        precisions.append(np.round(precision, 3))
        recalls.append(np.round(recall, 3))
    
    # return them to calling function
    return precisions, recalls

def pr_compute():
    # define thresholds from 0.2 to 0.65 with step size of 0.05
    thresholds = np.arange(start=0.2, stop=0.7, step=0.05)
    
    # call the compute_precision_recall function
    precisions, recalls = compute_precision_recall(
        yTrue=config.GROUND_TRUTH_PR, predScores=config.PREDICTION_PR,
        thresholds=thresholds,
    )
 
    # return the precisions and recalls
    return (precisions, recalls)

def plot_pr_curve(precisions, recalls, path):
    # plots the precision recall values for each threshold
    # and save the graph to disk
    plt.plot(recalls, precisions, linewidth=4, color="red")
    plt.xlabel("Recall", fontsize=12, fontweight='bold')
    plt.ylabel("Precision", fontsize=12, fontweight='bold')
    plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    plt.savefig(path)
    plt.show()