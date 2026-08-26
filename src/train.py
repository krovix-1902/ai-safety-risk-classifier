from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from risk_classifier import SimpleDecisionTree

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "synthetic_safety_dataset.csv")

feature_names = [c for c in df.columns if c != "risk"]
X = df[feature_names].values
labels = {"low": 0, "medium": 1, "high": 2}
y = df["risk"].map(labels).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

custom = SimpleDecisionTree(max_depth=3).fit(X_train, y_train)
custom_pred = custom.predict(X_test)

sklearn_tree = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=42)
sklearn_tree.fit(X_train, y_train)
sklearn_pred = sklearn_tree.predict(X_test)

print("=== Prototype AI Safety Risk Classifier ===")
print(f"Dataset size: {len(df)}")
print(f"Custom tree accuracy: {accuracy_score(y_test, custom_pred):.3f}")
print(f"sklearn accuracy:      {accuracy_score(y_test, sklearn_pred):.3f}")
print("\nCustom confusion matrix:")
print(confusion_matrix(y_test, custom_pred))
print("\nCustom classification report:")
print(classification_report(y_test, custom_pred, target_names=["low","medium","high"], zero_division=0))
print("\nLearned custom-tree rules:")
print(custom.export_rules(feature_names))
print("\nsklearn tree:")
print(export_text(sklearn_tree, feature_names=feature_names))
# Failure Analysis
print("\n=== Failure Analysis ===")

misclassified_idx = [
    i for i in range(len(y_test))
    if y_test[i] != custom_pred[i]
]

print(f"Misclassified: {len(misclassified_idx)} out of {len(y_test)}")

for idx in misclassified_idx:
    print(
        f"Sample {idx}: "
        f"true={y_test[idx]}, "
        f"predicted={custom_pred[idx]}, "
        f"features={X_test[idx]}"
    )