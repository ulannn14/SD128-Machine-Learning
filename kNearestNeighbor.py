import numpy as np

def compute_accuracy(true_labels, predicted_labels):
    """
    Computes the accuracy between true labels and predicted labels.
    """
    correct = np.sum(true_labels == predicted_labels)
    total = len(true_labels)
    return correct / total


class KNearestNeighbor:
    def __init__(self):
        self.train_data = None
        self.train_labels = None

    def train(self, data, labels):
        """
        Stores (already normalized) training data and labels.
        """
        # Cast to float64 to reduce rounding errors
        self.train_data = data.astype(np.float64)
        self.train_labels = labels

    def evaluate(self, data, labels, k=1):
        """
        Evaluates the kNN model on a given dataset using vectorized distance computation.
        """
        data = data.astype(np.float64)

        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2x·y
        test_sq = np.sum(np.square(data), axis=1)[:, np.newaxis]        # (num_test, 1)
        train_sq = np.sum(np.square(self.train_data), axis=1)[np.newaxis, :]  # (1, num_train)
        cross_term = np.dot(data, self.train_data.T)

        # Prevent tiny negatives due to floating-point error
        dists = np.sqrt(np.abs(test_sq + train_sq - 2 * cross_term))

        # # Optional: if evaluating on training data, avoid self-distance = 0
        # if data.shape == self.train_data.shape and np.allclose(data, self.train_data):
        #     np.fill_diagonal(dists, np.inf)

        # Indices of k nearest neighbors
        nn_idx = np.argsort(dists, axis=1)[:, :k]
        nn_labels = self.train_labels[nn_idx]

        # Majority vote
        preds = np.array([
            np.bincount(row.astype(int)).argmax() if np.issubdtype(row.dtype, np.integer)
            else np.unique(row, return_counts=True)[0][np.argmax(np.unique(row, return_counts=True)[1])]
            for row in nn_labels
        ])

        return preds
