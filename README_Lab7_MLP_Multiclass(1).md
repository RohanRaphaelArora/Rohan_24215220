# Lab #7: Keras MLP for Multiclass Classification

**Dataset used:** Wine dataset (built into scikit-learn) — 178 samples, 13 chemical
features, 3 classes (wine cultivars). Requires no download, so the notebook runs
end-to-end with zero setup.

---

## Objective
Implement a Multi-Layer Perceptron (MLP) using Keras/TensorFlow for a multiclass
classification problem, and analyze the effect of different **activation functions**
and **optimizers** on classification performance.

## Problem Statement
Predict which of 3 wine cultivars a sample belongs to, based on 13 chemical
measurements (alcohol content, malic acid, ash, flavanoids, color intensity, etc.).
A 3-class classification problem.

---

### Importing the libraries
NumPy, Pandas, Matplotlib, Seaborn for data handling/plotting; scikit-learn for the
dataset, splitting, scaling, and evaluation metrics; TensorFlow/Keras for the MLP
itself.

---

## 1. Dataset Preparation

### Load and explore the dataset
The Wine dataset is loaded via `sklearn.datasets.load_wine()` and converted to a
DataFrame for inspection — 178 samples, 13 numeric features, 3 target classes.

### Class distribution
A bar chart shows how many samples belong to each of the 3 wine cultivars, so we
can check the dataset isn't badly imbalanced.

### Handle missing values
The Wine dataset is clean by default, but a `df.isnull().sum()` check is included
as good practice.

### Encode the target variable
The integer class labels (0, 1, 2) are one-hot encoded using `to_categorical()` so
they match the shape of the Softmax output layer.

### Train/test split and feature scaling
An 80/20 stratified split keeps class proportions consistent between train and
test sets. `StandardScaler` is fit on the training data only (then applied to test
data) since the 13 features are on very different numeric scales.

---

## 2. MLP Model Development

**Architecture:** Input(13) → Dense(16, hidden activation) → Dense(8, hidden
activation) → Output Dense(3, Softmax)

- **Input features:** 13
- **Number of classes:** 3
- **Hidden layers:** 16 neurons, then 8 neurons
- **Output activation:** Softmax (multiclass probability distribution)
- **Loss function:** Categorical Crossentropy
- **Batch size:** 8
- **Epochs:** 50
- **Validation strategy:** 10% validation split during training

A reusable `build_model()` function takes the hidden activation and optimizer as
parameters, and a `run_experiment()` helper trains + evaluates a full config
(accuracy, precision, recall, F1) — used identically for both experiments below so
every model is compared fairly.

---

## 3. Experiment A - Hidden-Layer Activation Functions
Three models are trained with the optimizer fixed at **Adam**, varying only the
hidden-layer activation: **ReLU**, **Sigmoid**, **Tanh**. Training/validation
accuracy and loss vs. epoch are plotted for all three on the same axes for direct
comparison, followed by an explanation of why each activation function is
expected to behave differently (vanishing gradients, saturation, zero-centering).

## 4. Experiment B - Optimizer Comparison
Three models are trained with the hidden activation fixed at **ReLU**, varying
only the optimizer: **SGD**, **Adam**, **RMSprop**. Same plotting and comparison
approach as Experiment A, with an explanation of how each optimizer updates
weights differently (fixed learning rate vs. adaptive per-parameter rates).

---

## 5. Model Training
All 5 model configurations (`ReLU+Adam`, `Sigmoid+Adam`, `Tanh+Adam`, `ReLU+SGD`,
`ReLU+RMSprop`) are trained under identical conditions — same split, architecture,
epochs (50), batch size (8), and validation split (10%) — so differences in
results are attributable only to the varied activation/optimizer.

## 6. Model Evaluation
The best-performing model (highest test accuracy) is identified automatically,
retrained for a clean evaluation pass, and assessed with a full
`classification_report` (precision/recall/F1 per class) plus a confusion matrix
heatmap, with guidance on how to interpret which classes are most accurately
predicted vs. most confused.

## 7. Comparison Table
A pandas DataFrame summarizing all 5 experiments — activation, optimizer, test
accuracy, precision, recall, F1-score — in one table for side-by-side comparison,
matching the format requested in the assignment.

## 8. Analysis and Interpretation
A structured set of guided questions (best activation, best optimizer, best
combination, convergence speed, overfitting/underfitting signs, most
misclassified class, possible improvements) with pointers to exactly which table
column or plot answers each one — fill in your specific run's numbers here.

## 9. Final Model Selection
The model with the highest F1-score is selected automatically as a starting
suggestion, with a written justification covering why F1 (not just accuracy) was
used, convergence stability, and confusion matrix cleanliness.

---

## Observations

- With only 178 samples split 80/20, the test set has just ~36 samples, so accuracy/F1 can swing noticeably between runs — re-running the same config can shift test accuracy by a few percentage points, which is worth keeping in mind when comparing models that finish close together.
- **ReLU** typically converges fastest of the three activation functions, with training accuracy climbing sharply within the first 10–15 epochs, while **Sigmoid** tends to be the slowest to rise due to gradient saturation, and **Tanh** usually lands somewhere in between.
- **Adam** and **RMSprop** generally reach high training/validation accuracy within a similar number of epochs, both converging noticeably faster than **SGD**, which needs more epochs (or a higher learning rate/momentum) to catch up on this dataset.
- Because the 13 Wine features are on very different raw scales (e.g., `proline` in the hundreds vs. `hue` under 2), skipping `StandardScaler` would likely cause slower, less stable convergence — this is a bigger factor here than in the Iris lab, where feature ranges were already fairly close.
- The confusion matrix for the best model should show most errors, if any, between the two cultivar classes whose chemical profiles overlap most (commonly classes 1 and 2 in the Wine dataset), rather than errors spread evenly across all three classes.
- Validation accuracy tracking training accuracy closely (rather than plateauing far below it) across all 5 experiments would indicate the small architecture (16→8 neurons) is appropriately sized for this dataset rather than overfitting it.

*(Replace the general expectations above with your own notebook's actual printed numbers and plot shapes once you run it — e.g., exact final train/val accuracy per model, which class was most misclassified in your confusion matrix, and how many epochs each optimizer took to plateau.)*

---

## Conclusion
This lab implements a Keras/TensorFlow MLP for multiclass classification on the
Wine dataset, comparing three hidden-layer activation functions and three
optimizers under identical train/test conditions, and selecting a final model
using accuracy, precision, recall, F1-score, and confusion matrix analysis
together — rather than accuracy alone.

---

## How to Run
1. Open `Lab7_MLP_Multiclass_Wine.ipynb` in Google Colab or Jupyter Notebook.
2. Run all cells top to bottom — the dataset loads instantly (no download needed).
3. Training all 5 models (50 epochs each) takes well under a minute on CPU, since
   the dataset is small.
4. Review the printed comparison table, plots, and confusion matrix, and fill in
   the Analysis (Section 8) and Final Model Selection (Section 9) writeups with
   your own run's specific numbers.

## Requirements
- Python 3.x
- TensorFlow 2.x (`pip install tensorflow`)
- scikit-learn (`pip install scikit-learn`)
- pandas, matplotlib, seaborn (`pip install pandas matplotlib seaborn`)

## Dataset
[Wine recognition dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset)
— built into scikit-learn (`sklearn.datasets.load_wine`), originally from the UCI
Machine Learning Repository.
