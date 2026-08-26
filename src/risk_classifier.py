from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    value: Optional[int] = None

class SimpleDecisionTree:
    """Small educational binary decision tree using Gini impurity."""

    def __init__(self, max_depth: int = 3, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.classes_ = None

    @staticmethod
    def gini(y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)

    def _best_split(self, X, y):
        best_gain, best_feature, best_threshold = 0.0, None, None
        parent = self.gini(y)
        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                left = X[:, feature] <= threshold
                right = ~left
                if not left.any() or not right.any():
                    continue
                n = len(y)
                child = (left.sum()/n)*self.gini(y[left]) + (right.sum()/n)*self.gini(y[right])
                gain = parent - child
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold
        return best_feature, best_threshold, best_gain

    def _build(self, X, y, depth):
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or len(np.unique(y)) == 1):
            values, counts = np.unique(y, return_counts=True)
            return Node(value=int(values[np.argmax(counts)]))

        feature, threshold, gain = self._best_split(X, y)
        if feature is None or gain <= 0:
            values, counts = np.unique(y, return_counts=True)
            return Node(value=int(values[np.argmax(counts)]))

        left = X[:, feature] <= threshold
        return Node(
            feature=feature,
            threshold=threshold,
            left=self._build(X[left], y[left], depth + 1),
            right=self._build(X[~left], y[~left], depth + 1),
        )

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.root = self._build(np.asarray(X, dtype=float), np.asarray(y), 0)
        return self

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root) for x in X])

    def export_rules(self, feature_names, node=None, indent=""):
        node = self.root if node is None else node
        if node.value is not None:
            return indent + f"→ {self.classes_[node.value]}\n"
        text = indent + f"if {feature_names[node.feature]} <= {node.threshold}:\n"
        text += self.export_rules(feature_names, node.left, indent + "  ")
        text += indent + f"else ({feature_names[node.feature]} > {node.threshold}):\n"
        text += self.export_rules(feature_names, node.right, indent + "  ")
        return text
