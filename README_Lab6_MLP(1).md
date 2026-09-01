# Lab #6: Keras MLP for Regression - Diabetes Progression Predictor

## Lab #6: Implementing an MLP using TensorFlow/Keras for a Regression Problem

This lab builds a Multi-Layer Perceptron (MLP) using **TensorFlow/Keras** that
predicts a **continuous numeric value** — disease progression score — instead of
a class label. It is fully application oriented: run the notebook top to bottom
and it loads its own dataset, trains the model, and predicts a real value at the
end.

---

### Importing the libraries

We use `tensorflow`/Keras for the MLP itself, and `scikit-learn` for the dataset,
train/test split, feature scaling, and regression evaluation metrics (MSE, MAE,
R²).

---

## Part 0 - Loading the Dataset

Instead of downloading anything, the notebook loads the **Diabetes dataset**
built directly into scikit-learn (`load_diabetes()`) — 442 patient records with
10 baseline health measurements (age, BMI, blood pressure, blood serum levels,
etc.), and a continuous target measuring disease progression one year later.
This makes the notebook runnable anywhere, instantly, with zero setup.

---

## Part 1 - Data Preprocessing

### Checking for missing values
A quick `isnull().sum()` check confirms the dataset is already clean (it is, by
default).

### Train/test split and feature scaling
An 80/20 train/test split is used. `StandardScaler` is fit on the training data
only, then applied to both sets, since the 10 features are on different numeric
scales and MLPs train better when inputs are standardized.

---

## Part 2 - Building the MLP

For regression, the architecture looks almost identical to a classification MLP,
except the **output layer** has just **1 neuron with no activation function**
(linear output), since we want a raw numeric prediction rather than a class
probability.

### Initialising the model
A `Sequential` Keras model is created as the container for our MLP layers.

### Input Layer + First Hidden Layer
A `Dense` layer with 16 neurons and ReLU activation, taking in all 10 scaled
features.

### Second Hidden Layer
A second `Dense` layer with 8 neurons and ReLU activation, letting the network
learn more complex, non-linear combinations of the first layer's outputs.

### Output Layer (Regression)
A single `Dense` neuron with **no activation function** — this linear output is
what makes this a regression model instead of a classifier.

---

## Part 3 - Training the MLP

### Compiling the model
The model is compiled with:
- **Optimizer:** `adam` — how the model updates its weights.
- **Loss Function:** `mse` (Mean Squared Error) — the standard loss for
  regression, penalizing larger errors more heavily.
- **Metric:** `mae` (Mean Absolute Error) — easier to interpret since it's in
  the same units as the target.

### Training the model
The model is trained (`mlp.fit`) for 100 epochs with a 10% validation split, so
we can track how well it generalizes to unseen data during training.

### Plotting Training vs Validation Loss
A line plot of training and validation MSE over epochs, used to visually check
for overfitting (validation loss rising while training loss keeps falling) or
underfitting (both staying high).

---

## Part 4 - Evaluating the Model

The trained model is evaluated on the held-out test set using four standard
regression metrics: **MSE**, **RMSE** (easier to interpret, same units as the
target), **MAE**, and **R²** (how much of the variance in the target the model
explains — 1.0 is a perfect fit).

### Actual vs Predicted Plot
A scatter plot of actual vs. predicted values, with a red dashed diagonal line
representing a perfect prediction — points closer to that line indicate more
accurate predictions.

---

## Part 5 - Making a Single Prediction

This is the application-oriented step: one real patient's data is pulled from
the test set, fed through the trained model, and the predicted disease
progression score is compared directly against the actual value.

![alt text](image.png)
---

## Observations

- With only 442 samples split 80/20, the test set has just ~89 patients, so MSE/R² can shift noticeably between runs — re-running training with a different `random_state` can move R² by several hundredths, which is worth mentioning if your prof asks about run-to-run variability.
- Training loss (MSE) should drop sharply in the first ~20 epochs and then flatten out, while validation loss typically plateaus a bit higher and earlier — if validation loss starts rising again while training loss keeps falling, that's a sign of overfitting on a dataset this small.
- Because the 10 baseline features are on different raw scales (e.g., blood serum measurements vs. a 0/1-ish sex indicator), skipping `StandardScaler` would likely slow convergence and could bias the network toward whichever raw feature has the largest numeric range.
- R² on the Diabetes dataset with a small MLP like this commonly lands in the moderate range (roughly 0.3–0.5) rather than near 1.0 — this dataset has a lot of natural physiological noise, so a "perfect" fit isn't realistic or even desirable (it would suggest overfitting).
- On the Actual vs Predicted scatter plot, points should cluster loosely around the red diagonal line rather than sitting exactly on it; a wide, roughly symmetric scatter around the line indicates unbiased (not systematically over/under) predictions, while a curve or fan-shaped spread would suggest the model is missing some non-linear relationship or has uneven error across the target range.
- The single-prediction step (Part 5) confirms the model works end-to-end, but any individual patient's predicted-vs-actual gap can be larger than the average test MAE simply due to per-sample noise — the aggregate metrics (MSE/RMSE/MAE/R²) are the more reliable measure of overall performance than any one example.

---

## How to Run
1. Open `Lab6_MLP_Regression.ipynb` in Google Colab or Jupyter Notebook.
2. Run all cells in order, top to bottom.
3. The dataset loads instantly — no downloading or manual steps needed.
4. Training (100 epochs on 442 samples) takes well under a minute, even on CPU.
5. The final cells show the model's predicted vs. actual value for a real
   patient.

## Requirements
- Python 3.x
- TensorFlow 2.x (`pip install tensorflow`)
- scikit-learn (`pip install scikit-learn`)
- pandas, matplotlib (`pip install pandas matplotlib`)

## Dataset
[Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
— built into scikit-learn (`sklearn.datasets.load_diabetes`), a standard
regression benchmark dataset.
