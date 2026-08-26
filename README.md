# AI Safety Risk Classifier

> **Origin.** This began as a decision-tree implementation for a university AI
> coursework (Gini impurity and a depth-2 tree on the Car Evaluation dataset).
> I rebuilt it here as a standalone tested package and applied it to a different
> problem: risk triage of AI-system interactions.

A small, interpretable machine-learning prototype for **risk triage of AI-system interactions**.

## Research question

> Can a simple, interpretable decision tree classify potential AI safety risks
> from a small set of observable interaction features?

## What this project does

Classifies an interaction as **low**, **medium**, or **high** risk using seven binary features:

- harmful intent
- actionable instructions
- critical-system targeting
- sensitive-data involvement
- evasion language
- autonomy requests
- security-bypass requests

The decision tree is implemented **from scratch** — Gini impurity, best-split
selection, recursive tree construction and prediction — and compared against a
scikit-learn baseline.

## Results

| Model              | Test accuracy |
|--------------------|---------------|
| Custom tree (d=3)  | __%           |
| scikit-learn (d=3) | __%           |

Misclassified __ of __ held-out examples.

Because the labelling rule is hand-designed, this measures whether the tree can
recover a rule I encoded — not whether it generalises to real AI-safety data.

## Important limitation

This is a **research/learning prototype**, not a production safety system. The
dataset is synthetic, small, and hand-labelled by me. Results should not be read
as evidence that the classifier is reliable for real-world AI safety decisions.

A meaningful extension would replace the synthetic data with independently
labelled evaluation examples and measure robustness across different models,
prompt styles and annotators.

## Run

```bash
pip install -r requirements.txt
python src/train.py
pytest
```

## Next steps

1. Vary `max_depth` and compare training/test performance.
2. Ablate one feature at a time and observe the effect.
3. Replace synthetic labels with independently annotated examples.
4. Analyse false positives and false negatives rather than headline accuracy.