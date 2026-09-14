# AI/ML Final Project — ANN vs. SVM on the Adult Census Income Dataset

**Type:** Team project
**Contributors:** Carter Ward, Boyd Emmons
**Course:** CS 430-1 (Machine Learning) — Final Project
**Completed:** 12/07/2025

## Purpose
Final project for CS 430 (AI/ML): build and evaluate ML models on a real dataset and understand how they work internally, not just call a library and report a number. Boyd and I trained a feedforward ANN and a linear SVM on the UCI Adult Census Income dataset to predict whether a person earns more or less than $50K/year, implementing both models from scratch in NumPy instead of using `sklearn`/`keras` estimators.

## Problem and Approach
Binary classification on the UCI Adult dataset (`adult.data` train / `adult.test` test): predict `<=50K` vs. `>50K` income from 14 attributes (age, workclass, education, occupation, hours worked, capital gain/loss, native country, etc.). Pipeline in `FinalProject.py`:
- Load both files with correct column names, treat `"?"` as missing, strip trailing periods from test labels.
- Impute missing numeric fields with median, categorical with mode.
- Map income label to 0/1; one-hot encode the nine categorical columns (encoder fit on train, reused on test).
- Standardize all features (`StandardScaler`, fit on train only); hold out a stratified 20% validation split.
- Train `ManualANN` and `ManualLinearSVM`, written by hand in NumPy — no `sklearn.svm`/`keras`/`torch` for the math.
- Evaluate on the untouched test set: accuracy, precision, recall, F1, confusion matrix.

## Structure and Methodologies
- **Dependencies:** `numpy`, `pandas`, `matplotlib`, `scikit-learn` (only for `train_test_split`, `StandardScaler`, `OneHotEncoder`, and metrics — not the models). Python 3.13.9, VS Code.
- Single script, `FinalProject.py`, sectioned into: data loading/cleaning (`load_dataset`, `handle_missing_values`, `split_features_labels`), feature engineering (`encode_categorical`, `scale_features`), models, evaluation/reporting, and `main()`.
- `ManualANN`: He-initialized weights, ReLU hidden layers, sigmoid output, binary cross-entropy loss (+L2), hand-derived backprop for mini-batch gradient descent.
- `ManualLinearSVM`: primal linear SVM with vectorized hinge-loss gradient updates to `w`/`b` from a `0.5*||w||^2 + C*hinge_loss` objective.
- `evaluate_models`, `save_model_comparison` (writes `model_comparison_report.txt`), `plot_ann_convergence`/`plot_svm_convergence` (loss curve PNGs).

## Process
1. Loaded raw data: 32,561 training rows, 16,281 test rows, 15 columns each; verified shapes/dtypes against `adult.names`.
2. Checked missing values: gaps only in `workclass`, `occupation`, `native-country` (4,262 train / 2,203 test cells); class balance ~76% `<=50K` / 24% `>50K`.
3. Imputed missing values (median/mode) and confirmed zero remaining gaps.
4. Split features/label, mapped income to 0/1.
5. One-hot encoded categorical columns, fitting only on train.
6. Standardized full feature matrix (cast to `float32`).
7. Carved a stratified 80/20 train/validation split.
8. Trained the SVM first: 15 epochs, full-batch hinge-loss gradient descent, `C=1.0`, lr `1e-3` — loss oscillated (181,220 → 41,656 → 67,630 → 59,771), indicating an overly aggressive fixed learning rate (see `svm_convergence.png`).
9. Trained the ANN: 2 hidden layers (32, 16 units, ReLU, He init), 10 epochs, batch size 256, lr `5e-3` — loss rose (0.6705 → 0.6963 → 0.7679), pointing to too-high a learning rate (see `ann_convergence.png`).
10. Evaluated both models on the test set.
11. Saved `model_comparison_report.txt`, preprocessed train/test CSVs, and both convergence PNGs.

## Outcome
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| SVM   | 0.7996   | 0.5513    | 0.8144 | 0.6575 |
| ANN   | 0.8143   | 0.6438    | 0.4789 | 0.5493 |

The ANN beat the SVM on accuracy (0.8143 vs. 0.7996), but the two trade off: the SVM catches far more true `>50K` earners (recall 0.81 vs. 0.48) at the cost of more false positives (2,549 vs. 1,019), while the ANN is more precise but misses over half of true `>50K` cases (2,004 false negatives). Given the ~76/24 class imbalance, accuracy alone is misleading — which is why precision/recall/F1/confusion matrices matter. The project demonstrates a full ML pipeline (cleaning, encoding, scaling, training, evaluating) built and debugged from scratch, and hands-on understanding of backprop, hinge-loss SVM optimization, and why accuracy alone doesn't tell the whole story on imbalanced data.

**How to run:**
```
pip install numpy pandas scikit-learn matplotlib
python FinalProject.py
```
Requires `adult.data`/`adult.test` in the same folder; takes ~10 minutes on a laptop (pure NumPy, CPU).
