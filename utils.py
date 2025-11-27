
import numpy as np

def compute_accuracy(true_labels, predicted_labels):
    """
    Computes the accuracy between true labels and predicted labels.
    """
    correct = np.sum(true_labels == predicted_labels)
    total = len(true_labels)
    return correct / total