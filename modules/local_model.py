"""
Module 2: Local Comment Classifier & Multi-Class Evaluation Metrics.

Implements:
1. CommentClassifierMLP: Multi-class neural network (V -> 64 -> 32 -> 4) in pure NumPy
   (zero DLL blocks, laptop-friendly, fully differentiable for Flower FedAvg/FedProx).
2. Softmax cross-entropy loss with optional FedProx proximal penalty.
3. Multi-class metrics: macro-F1, Precision, Recall, ROC-AUC, Confusion Matrix, and
   False Positive Rate (FPR) on benign comments.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np

LABEL_NAMES = ["Positive", "Neutral", "Criticism", "Abusive"]

class CommentClassifierMLP:
    """
    Feedforward Neural Network for 4-Class Comment Classification:
    0: Positive, 1: Neutral, 2: Negative-but-not-abusive (Criticism), 3: Abusive-Vulgar.
    Architecture: Input (V=1200) -> Dense(64) -> ReLU -> Dense(32) -> ReLU -> Dense(4) -> Softmax.
    """
    def __init__(self, input_dim: int = 1200, hidden1: int = 64, hidden2: int = 32, num_classes: int = 4, seed: int = 42):
        np.random.seed(seed)
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.num_classes = num_classes

        # He initialization
        self.W1 = np.random.randn(input_dim, hidden1).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.W2 = np.random.randn(hidden1, hidden2).astype(np.float32) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2, dtype=np.float32)
        self.W3 = np.random.randn(hidden2, num_classes).astype(np.float32) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(num_classes, dtype=np.float32)

        # Adam optimizer parameters
        self.m = [np.zeros_like(p) for p in self.get_weights()]
        self.v = [np.zeros_like(p) for p in self.get_weights()]
        self.t = 0

    def get_weights(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_weights(self, weights: List[np.ndarray]):
        self.W1 = np.asarray(weights[0], dtype=np.float32).copy()
        self.b1 = np.asarray(weights[1], dtype=np.float32).copy()
        self.W2 = np.asarray(weights[2], dtype=np.float32).copy()
        self.b2 = np.asarray(weights[3], dtype=np.float32).copy()
        self.W3 = np.asarray(weights[4], dtype=np.float32).copy()
        self.b3 = np.asarray(weights[5], dtype=np.float32).copy()

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities via Softmax."""
        X = np.asarray(X, dtype=np.float32)
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)
        z3 = a2 @ self.W3 + self.b3

        # Numerically stable Softmax
        exp_z = np.exp(z3 - np.max(z3, axis=1, keepdims=True))
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class index (0, 1, 2, 3)."""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def predict_abuse_prob(self, X: np.ndarray) -> np.ndarray:
        """Return probability of comment being abusive (class 3)."""
        probs = self.forward(X)
        return probs[:, 3]

    def train_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.005,
        proximal_mu: float = 0.0,
        global_weights: Optional[List[np.ndarray]] = None
    ) -> float:
        """Execute one Adam-optimized gradient descent step with optional FedProx proximal penalty."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).flatten()
        m = len(X)
        if m == 0:
            return 0.0

        # Forward pass
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)
        z3 = a2 @ self.W3 + self.b3

        exp_z = np.exp(z3 - np.max(z3, axis=1, keepdims=True))
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        # Cross-Entropy Loss
        log_probs = np.log(probs[np.arange(m), y] + 1e-12)
        loss = float(-np.mean(log_probs))

        # Backward gradients: dL/dz3 = (probs - one_hot) / m
        Y_one_hot = np.zeros_like(probs)
        Y_one_hot[np.arange(m), y] = 1.0
        dz3 = (probs - Y_one_hot) / m

        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        grads = [dW1, db1, dW2, db2, dW3, db3]

        # FedProx proximal regularization: mu * (w - w_global)
        if proximal_mu > 0.0 and global_weights is not None:
            curr_weights = self.get_weights()
            for i in range(len(grads)):
                grads[i] += proximal_mu * (curr_weights[i] - global_weights[i])

        # Adam optimization update
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        weights = self.get_weights()
        for i in range(len(weights)):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grads[i]
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grads[i] ** 2)
            m_hat = self.m[i] / (1.0 - beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - beta2 ** self.t)
            weights[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        self.set_weights(weights)
        return loss

# Aliases for Flower FL compatibility
EngagementMLP = CommentClassifierMLP

def get_model_parameters(model: CommentClassifierMLP) -> List[np.ndarray]:
    return model.get_weights()

def set_model_parameters(model: CommentClassifierMLP, parameters: List[np.ndarray]):
    model.set_weights(parameters)

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Computes Precision, Recall, Macro-F1, Accuracy, Abuse-specific metrics,
    Confusion Matrix (4x4), and False-Positive Rate on non-abusive comments.
    """
    y_true = np.asarray(y_true, dtype=np.int64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.int64).flatten()
    n = len(y_true)

    # 4x4 Confusion Matrix: rows = true, cols = predicted
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    accuracy = float(np.mean(y_true == y_pred))

    # Per-class metrics
    precisions = []
    recalls = []
    f1s = []

    for c in range(4):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp

        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f = (2 * p * r) / max(1e-12, p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    # Abuse class specific metrics (Class 3 = Abusive-Vulgar)
    abuse_prec = float(precisions[3])
    abuse_rec = float(recalls[3])
    abuse_f1 = float(f1s[3])

    # False Positive Rate on normal/civil users:
    # Comments that are non-abusive (classes 0, 1, 2) mistakenly classified as abusive (class 3)
    non_abusive_total = int(np.sum(cm[:3, :]))
    false_abusive_alarms = int(np.sum(cm[:3, 3]))
    fpr_normal = float(false_abusive_alarms / max(1, non_abusive_total))

    # Simple ROC-AUC approximation for binary abuse detection
    roc_auc = 0.50
    if y_prob is not None:
        y_binary = (y_true == 3).astype(int)
        abuse_scores = np.asarray(y_prob, dtype=np.float32)
        # Approximate AUC via rank-sum Wilcoxon statistic
        pos_scores = abuse_scores[y_binary == 1]
        neg_scores = abuse_scores[y_binary == 0]
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            # Random sub-sample for speed if large
            sub_pos = pos_scores if len(pos_scores) < 1000 else np.random.choice(pos_scores, 1000, replace=False)
            sub_neg = neg_scores if len(neg_scores) < 1000 else np.random.choice(neg_scores, 1000, replace=False)
            pairs = np.sum(sub_pos[:, None] > sub_neg[None, :]) + 0.5 * np.sum(sub_pos[:, None] == sub_neg[None, :])
            roc_auc = float(pairs / (len(sub_pos) * len(sub_neg)))

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "abuse_precision": round(abuse_prec, 4),
        "abuse_recall": round(abuse_rec, 4),
        "abuse_f1": round(abuse_f1, 4),
        "fpr_normal_users": round(fpr_normal, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm.tolist()
    }

def train_classifier_epoch(
    model: CommentClassifierMLP,
    train_loader: Any,
    global_params: Optional[List[np.ndarray]] = None,
    proximal_mu: float = 0.0,
    lr: float = 0.005
) -> float:
    """Train classifier across batches for one epoch."""
    losses = []
    for X_batch, y_batch in train_loader:
        loss = model.train_step(
            X=X_batch,
            y=y_batch,
            lr=lr,
            proximal_mu=proximal_mu,
            global_weights=global_params
        )
        losses.append(loss)
    return float(np.mean(losses)) if losses else 0.0

def evaluate_classifier(
    model: CommentClassifierMLP,
    data_loader: Any
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate classifier on test DataLoader, returning metrics, targets, predictions, and abuse probabilities."""
    all_y = []
    all_preds = []
    all_probs = []

    for X_batch, y_batch in data_loader:
        probs = model.forward(X_batch)
        preds = np.argmax(probs, axis=1)
        all_y.extend(np.asarray(y_batch).flatten())
        all_preds.extend(preds)
        all_probs.extend(probs[:, 3])  # abuse probability

    y_true = np.array(all_y, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)
    y_prob = np.array(all_probs, dtype=np.float32)

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    return metrics, y_true, y_pred, y_prob
