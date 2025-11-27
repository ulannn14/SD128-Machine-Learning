import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# -----------------------------
# 1. Accuracy
# -----------------------------
def compute_accuracy(true_labels, predicted_labels):
    correct = np.sum(true_labels == predicted_labels)
    total = len(true_labels)
    return correct / total


# -----------------------------
# 2. Confusion Matrix (first N classes)
# -----------------------------
def plot_confusion(y_true, y_pred, class_map, title="Confusion Matrix", max_classes=20):
    cm = confusion_matrix(y_true, y_pred)
    class_labels = [class_map[i] for i in range(max_classes)]

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm[:max_classes, :max_classes],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()


# -----------------------------
# 3. Misclassified Samples
# -----------------------------
def visualize_misclassifications(X, y_true, y_pred, class_map, num_samples=5):
    misclassified_idxs = np.where(y_true != y_pred)[0]
    if len(misclassified_idxs) == 0:
        print("No misclassifications to show.")
        return

    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(misclassified_idxs[:num_samples]):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[idx].astype(np.uint8))
        true_class = class_map[y_true[idx]]
        pred_class = class_map[y_pred[idx]]
        plt.title(f"True: {true_class}\nPred: {pred_class}", fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 4. Training History Plot
# -----------------------------
def plot_history(history, title_prefix="Model"):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Graph
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.title(f"{title_prefix} Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss Graph
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title(f"{title_prefix} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()
