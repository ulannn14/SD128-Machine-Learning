import numpy as np
from utils import compute_accuracy 

# ---------- Activations ----------
def relu(x): 
    return np.maximum(0, x)

def relu_grad(x): 
    return (x > 0).astype(x.dtype)

def sigmoid(x):
    pos = x >= 0
    neg = ~pos
    out = np.empty_like(x)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)
    return out

def sigmoid_grad(sig_x): 
    return sig_x * (1.0 - sig_x)

def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (np.sum(ez, axis=1, keepdims=True) + 1e-12)

def one_hot(y, num_classes):
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


# ---------- Model ----------
class MultiLayerNN:
    """
    Fully connected NN:
        Input -> Hidden(1..n) -> Softmax
    Uses same naming style as KNearestNeighbor:
        - .train() to train the model
        - .evaluate() to test/validate
    """

    def __init__(self, layer_sizes, activation="relu", l2=0.0, seed=42, weight_scale=None):
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.act_name = activation.lower()
        self.l2 = float(l2)

        rng = np.random.default_rng(seed)
        self.weights, self.biases = [], []

        for i in range(self.L):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            if weight_scale is not None:
                W = rng.normal(0, weight_scale, size=(fan_in, fan_out))
            else:
                if self.act_name == "relu":
                    W = rng.normal(0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
                else:
                    W = rng.normal(0, np.sqrt(1.0 / fan_in), size=(fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.weights.append(W.astype(np.float32))
            self.biases.append(b.astype(np.float32))


    # ----- forward -----
    def _forward(self, X):
        A = X
        caches = []
        for i in range(self.L - 1):
            Z = A @ self.weights[i] + self.biases[i]
            if self.act_name == "relu":
                A_next = relu(Z)
                cache = (A, Z)
            else:
                A_next = sigmoid(Z)
                cache = (A, A_next)
            caches.append(cache)
            A = A_next

        logits = A @ self.weights[-1] + self.biases[-1]
        probs = softmax(logits)
        caches.append((A, logits))
        return probs, caches


    # ----- loss + grads -----
    def _loss_and_grads(self, X, y):
        B = X.shape[0]
        probs, caches = self._forward(X)
        Y = one_hot(y, self.layer_sizes[-1])

        eps = 1e-9
        ce = -np.sum(Y * np.log(np.clip(probs, eps, 1.0))) / B
        l2_term = 0.5 * self.l2 * sum(np.sum(W * W) for W in self.weights)
        loss = ce + l2_term

        grads_W, grads_b = [None]*self.L, [None]*self.L
        dZ = (probs - Y) / B

        # Output layer
        A_prev, _ = caches[-1]
        grads_W[-1] = A_prev.T @ dZ + self.l2 * self.weights[-1]
        grads_b[-1] = np.sum(dZ, axis=0, keepdims=True)
        dA = dZ @ self.weights[-1].T

        # Hidden layers (reverse)
        for i in reversed(range(self.L - 1)):
            A_prev, Z_or_A = caches[i]
            if self.act_name == "relu":
                dZ = dA * relu_grad(Z_or_A)
            else:
                dZ = dA * sigmoid_grad(Z_or_A)
            grads_W[i] = A_prev.T @ dZ + self.l2 * self.weights[i]
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True)
            dA = dZ @ self.weights[i].T

        return loss, grads_W, grads_b

    # ----- TRAIN -----
    def train(self, data, labels, val_data=None, val_labels=None,
              epochs=20, lr=1e-3, batch_size=128, shuffle=True, verbose=True):

        N = data.shape[0]
        train_losses = []
        val_accuracies = []

        for epoch in range(1, epochs + 1):
            if shuffle:
                idx = np.random.permutation(N)
                data, labels = data[idx], labels[idx]

            total_loss = 0.0
            for i in range(0, N, batch_size):
                xb = data[i:i+batch_size]
                yb = labels[i:i+batch_size]
                loss, grads_W, grads_b = self._loss_and_grads(xb, yb)
                total_loss += loss * xb.shape[0]

                # SGD update
                for j in range(self.L):
                    self.weights[j] -= lr * grads_W[j]
                    self.biases[j] -= lr * grads_b[j]

            avg_loss = total_loss / N
            train_losses.append(avg_loss)

            if val_data is not None and val_labels is not None:
                preds = self.evaluate(val_data)
                acc = compute_accuracy(val_labels, preds)
                val_accuracies.append(acc)
                if verbose:
                    print(f"Epoch {epoch:3d} | loss {avg_loss:.4f} | accuracy {acc*100:5.2f}%")
            else:
                if verbose:
                    print(f"Epoch {epoch:3d} | loss {avg_loss:.4f}")

        # return collected history for plotting / tuning
        return {
            "train_loss": train_losses,
            "val_acc": val_accuracies
        }

    # ----- EVALUATE -----
    def evaluate(self, data, batch_size=2048):
        N = data.shape[0]
        preds = np.empty(N, dtype=np.int64)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            probs, _ = self._forward(data[start:end])
            preds[start:end] = np.argmax(probs, axis=1)
        return preds