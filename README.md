# AI-MLProj: Predicting Income from the Adult Census Dataset

A CS 430 (Machine Learning) final project by Carter Ward and Boyd Emmons. The project trains and compares two classic supervised learning models — a neural network and a support vector machine — to predict whether a person's income exceeds $50K/year, using the classic UCI "Adult" (Census Income) dataset. Both models are implemented from scratch in NumPy rather than imported from a library, so the project doubles as a from-first-principles study of how a neural network and an SVM actually learn.

## 1. Purpose

This project exists to answer a simple question with real data: can we predict, from a handful of demographic and employment attributes, whether someone earns more than $50,000 a year? That's the well-known "Adult" / Census Income prediction task, a standard benchmark for binary classification in machine learning.

Beyond the prediction task itself, the deeper purpose of the project was educational. Rather than calling `sklearn.neural_network.MLPClassifier` or `sklearn.svm.SVC` and treating the models as black boxes, we wrote our own Artificial Neural Network (`ManualANN`) and our own Linear Support Vector Machine (`ManualLinearSVM`) using only NumPy for the math. That choice forced us to understand — and implement by hand — forward propagation, backpropagation, gradient descent, and hinge-loss optimization, instead of just calling `.fit()` on someone else's implementation. The project was built to demonstrate that understanding, not just to produce a working classifier.

## 2. Problem and approach

**The problem:** the Adult dataset (`adult.data` / `adult.test`, sourced from the U.S. Census Bureau via the UCI Machine Learning Repository) contains 48,842 records with 14 attributes each — age, workclass, education, marital status, occupation, relationship, race, sex, capital gain/loss, hours worked per week, and native country — labeled with whether that person's income is `<=50K` or `>50K`. It's an imbalanced classification problem: roughly 76% of records fall in the `<=50K` class and only 24% in `>50K`, which makes accuracy alone a misleading metric and precision/recall/F1 important to track.

**Our approach** was to build a complete, reproducible ML pipeline in a single well-organized script (`FinalProject.py`) rather than a notebook, and to implement both a linear-algebra-based neural network and a linear SVM ourselves so their inner workings would be transparent:

- Load the raw `adult.data` (train) and `adult.test` (test) files, applying the known column schema from `adult.names` and treating `"?"` as missing data.
- Clean the test-set labels (they carry a trailing period in the raw file, e.g. `>50K.`) so train and test labels match.
- Impute missing values — median for numeric columns, mode for categorical columns.
- One-hot encode all categorical features and standardize (z-score) all numeric features, fitting the encoder/scaler on the training set only and re-using them on the test set to avoid data leakage.
- Split off a stratified 20% validation set from the training data so both models could be monitored for overfitting during training.
- Train a hand-built `ManualLinearSVM` (hinge-loss, mini-batch-free vectorized gradient descent) and a hand-built `ManualANN` (a small feedforward network with ReLU hidden layers, a sigmoid output, and manual backpropagation) on the same preprocessed features.
- Evaluate both models on the held-out test set with accuracy, precision, recall, F1, and a confusion matrix, then write a human-readable comparison report and convergence plots.

## 3. Structure and methodologies

**Language & core libraries** (all visible in `FinalProject.py`'s imports):
- **Python 3** as the implementation language.
- **NumPy** — used to implement the ANN's forward pass, backpropagation, and weight updates, and the SVM's hinge-loss gradient, entirely by hand (no autodiff, no deep learning framework).
- **pandas** — for loading, cleaning, and exploring the tabular census data (`load_dataset`, `explore_dataset`, `handle_missing_values`).
- **scikit-learn** — used only for the well-understood, "solved" preprocessing and evaluation utilities, not for the models themselves: `train_test_split` (stratified validation split), `StandardScaler` (feature scaling), `OneHotEncoder` (categorical encoding), and the metrics module (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`).
- **Matplotlib** — to render training/validation loss curves (`ann_convergence.png`, `svm_convergence.png`).
- **`logging`** (Python standard library) — the entire pipeline logs its progress (dataset shape, class balance, missing-value counts, per-epoch training loss, final metrics) instead of using ad hoc `print()` statements, and that log is captured verbatim in `terminal_output.txt`.

**Key data structures / classes:**
- `ManualANN` — a configurable feedforward network (default architecture: input → 32 → 16 → 1) storing its weights and biases as lists of NumPy arrays, with `_forward`, `_backward`, `_compute_loss` (binary cross-entropy with optional L2 regularization), `fit` (mini-batch gradient descent), and `predict`/`predict_proba` methods.
- `ManualLinearSVM` — a primal linear SVM storing a weight vector `w` and bias `b`, trained with vectorized hinge-loss gradient descent (`fit`), with `decision_function` and `predict` methods.
- Plain functions for each pipeline stage (`load_dataset`, `encode_categorical`, `scale_features`, `evaluate_models`, `save_model_comparison`, `plot_ann_convergence`, `plot_svm_convergence`), keeping the script organized as a linear, readable pipeline rather than a monolithic `main()`.

**Data and artifacts included in the repo:**
- `adult.data`, `adult.test`, `adult.names`, `old.adult.names`, `Index` — the original UCI Adult dataset files and their documentation.
- `adult_preprocessed_train.csv`, `adult_preprocessed_test.csv` — the fully encoded/scaled feature matrices the pipeline produces, saved for inspection/reuse.
- `model_comparison_report.txt` — the generated metrics report (see Outcome below).
- `ann_convergence.png`, `svm_convergence.png` — training/validation loss curves for each model.
- `terminal_output.txt` — a full captured log of an actual pipeline run.
- `README.docx` — the original project write-up/report submitted for the course.

## 4. Process

Reconstructing the build from the code and the header comment in `FinalProject.py` ("Authors: Carter Ward, Boyd Emmons — Course: CS 430-1 — Date: 12/7/25 (Final Version)"), the project came together in stages typical of an end-to-end ML assignment:

1. **Data acquisition and exploration.** The Adult Census dataset was pulled in with its original file layout (`adult.data`/`adult.test`/`adult.names`/`Index`), and an exploratory pass (`explore_dataset`) was written first to check shape, dtypes, missing-value counts, and class balance — logged output shows 32,561 training rows and 16,281 test rows, with roughly 4,262 and 2,203 missing values respectively, concentrated in `workclass`, `occupation`, and `native-country`.
2. **Data cleaning and preprocessing.** Once the shape of the problem was understood, missing-value handling (median/mode imputation), label cleanup (stripping the trailing period from test-set income labels), one-hot encoding, and standardization were built out as separate, testable functions rather than inline script logic — a sign of iterating from a "just make it work" script toward a more maintainable one.
3. **Implementing the models from scratch.** Instead of using `sklearn`'s built-in classifiers, the team chose to hand-roll both a neural network and a linear SVM in NumPy. This was clearly the technical core of the assignment — the ANN required implementing He initialization, ReLU/sigmoid activations, binary cross-entropy loss, and full backpropagation by hand; the SVM required implementing the hinge-loss subgradient and a from-scratch training loop. Both models track per-epoch training and validation loss internally to support later diagnostics.
4. **Training and iteration.** Both models were trained with a stratified 80/20 train/validation split so the team could watch for overfitting. The SVM was trained for 15 epochs and the ANN for 10, both logging progress every few epochs — visible directly in `terminal_output.txt`, which captures an actual run end-to-end.
5. **Evaluation and reporting.** A dedicated evaluation stage (`evaluate_models`) computed accuracy, precision, recall, F1, and confusion matrices for both models on the untouched test set, and `save_model_comparison` wrote those results out to `model_comparison_report.txt` in a clean, readable format, including an automatic head-to-head comparison sentence.
6. **Visualization and artifact generation.** Finally, convergence plots for both models were generated with Matplotlib and saved as PNGs, and the fully preprocessed train/test matrices were exported as CSVs — turning the project from "a script that prints numbers" into a repository with durable, inspectable evidence of what the pipeline actually did.

The presence of both a `README.docx` (a formal written report, likely for course submission) and a captured `terminal_output.txt` log suggests the team ran the pipeline for real, verified its output, and documented results formally rather than just eyeballing print statements.

## 5. Outcome

The pipeline runs end-to-end and produces real, measured results, captured in `model_comparison_report.txt` and corroborated by `terminal_output.txt`:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| SVM (hand-built, linear, hinge loss)  | 0.7996 | 0.5513 | 0.8144 | 0.6575 |
| ANN (hand-built, 2 hidden layers)     | 0.8143 | 0.6438 | 0.4789 | 0.5493 |

The from-scratch ANN edged out the from-scratch SVM in overall accuracy (81.4% vs. 80.0%, a 1.47-point gap), while the SVM achieved substantially higher recall on the minority `>50K` class (81.4% vs. 47.9%) at the cost of more false positives. That trade-off is a genuine and interpretable finding: the SVM's hinge-loss objective and simple linear decision boundary catch more high-income individuals but with lower precision, while the ANN's extra capacity yields a better-calibrated but more conservative classifier for the minority class. Both models comfortably beat the 76% "always predict `<=50K`" baseline implied by the class distribution, confirming the features carry real predictive signal that both models learned to exploit.

Beyond the numbers, building this project exercised a full stack of practical machine learning skills:

- **Implementing gradient-based learning by hand** — writing forward/backward passes and a hinge-loss gradient in NumPy, rather than relying on a framework's autodiff, forced a concrete, working understanding of how backpropagation and SVM optimization actually update parameters.
- **Correct ML methodology** — fitting the encoder and scaler only on training data, using a stratified validation split, and evaluating exclusively on a held-out test set are all practices that prevent data leakage and inflated performance claims.
- **Handling real-world messy data** — missing values encoded as `"?"`, inconsistent label formatting between train/test files, and mixed categorical/numeric columns all had to be identified and handled explicitly.
- **Evaluating classifiers responsibly on imbalanced data** — recognizing that accuracy alone is insufficient on a ~76/24 class split, and reporting precision, recall, F1, and confusion matrices instead.
- **Producing reproducible, inspectable results** — logging every pipeline stage, saving processed datasets, and generating convergence plots turned the project into something whose claims can be checked, not just taken on faith.

Overall, the project demonstrates the ability to take a real dataset from raw files to a documented, evaluated, and compared pair of working classifiers — including the harder path of implementing the learning algorithms manually rather than leaning entirely on library defaults, which is exactly the kind of foundational understanding a machine learning course is meant to build.
