import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from risk_classifier import SimpleDecisionTree

def test_gini_pure_node():
    assert SimpleDecisionTree.gini(np.array([1,1,1])) == 0.0

def test_gini_mixed_node():
    assert abs(SimpleDecisionTree.gini(np.array([0,1])) - 0.5) < 1e-9

def test_tree_can_fit_simple_data():
    X = np.array([[0],[0],[1],[1]])
    y = np.array([0,0,1,1])
    model = SimpleDecisionTree(max_depth=2).fit(X,y)
    assert (model.predict(X) == y).all()
