# Perceptron from Scratch using NumPy

A simple Python implementation of a **single-layer Perceptron** built from scratch using only NumPy — no machine learning libraries involved. This project demonstrates the fundamentals of how the earliest neural network model learns to classify data.

## 📌 Overview

A **Perceptron** is the simplest form of an artificial neural network — a single neuron that:

1. Takes weighted inputs
2. Sums them together (plus a bias term)
3. Applies a **step activation function**
4. Outputs a binary result (0 or 1)

It learns using the classic **Perceptron Learning Rule**:

```
w = w + η · (y − ŷ) · x
```

Where:
- `w` = weights
- `η` (eta) = learning rate
- `y` = actual label
- `ŷ` = predicted label
- `x` = input features

## 🧠 What This Project Does

This notebook implements a `Perceptron` class from scratch and applies it to a real-world-style classification problem: **predicting whether a student will Pass or Fail** based on two features — **study hours** and **attendance percentage**.

Since this dataset is linearly separable, the perceptron successfully learns a decision boundary that perfectly classifies all training examples and generalizes correctly to new, unseen data.

## 📂 Files

| File | Description |
|------|-------------|
| `Prac_1.ipynb` | Jupyter notebook containing the full implementation, training, and predictions |

## ⚙️ How It Works

### 1. Activation Function — Step Function

```python
def step_function(self, z):
    return 1 if z >= 0 else 0
```

Outputs a hard 0 or 1 based on whether the weighted sum crosses zero. This is the classic activation used in Rosenblatt's original 1958 Perceptron (not ReLU or sigmoid).

### 2. Prediction

```python
def predict(self, x):
    z = np.dot(x, self.weights) + self.bias
    return self.step_function(z)
```

Computes the weighted sum of inputs plus bias, then passes it through the step function.

### 3. Training

The model iterates over the dataset for a set number of epochs, updating weights and bias after every misclassified example using the perceptron learning rule. Training stops early if a full epoch passes with zero errors (convergence).

## 📊 Dataset

**Student Pass/Fail Prediction**

| Feature | Description |
|---------|-------------|
| Study Hours | Number of hours studied |
| Attendance % | Class attendance percentage |
| Label | 0 = Fail, 1 = Pass |

```python
X_student = np.array([
    [1, 40], [2, 45], [2, 50], [3, 55], [4, 60],
    [5, 65], [6, 70], [7, 80], [8, 85], [9, 90],
    [3, 30], [1, 20]
])

y_student = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
```

Features are scaled (divided by their max values) before training to help the perceptron converge faster and more reliably.

## ▶️ How to Run

1. Clone this repository
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. Install dependencies
   ```bash
   pip install numpy jupyter
   ```
3. Launch the notebook
   ```bash
   jupyter notebook Prac_1.ipynb
   ```
4. Run all cells to see training progress and predictions.

## 📈 Sample Output

```
Training on Student Pass/Fail Data:
Epoch   Weights           Bias      Errors
------------------------------------------------------------
1       [ 0.011 -0.006]   -0.100    3
2       [0.033 0.033]     -0.100    2
...
10      [0.156 0.161]     -0.200    0

Converged early!

New student (6 hrs study, 75% attendance) → Pass
```

The model converged in 10 epochs and correctly classified all 12 training samples, then correctly predicted the outcome for a brand-new, unseen student.

## 🎯 Key Learnings

- How a single artificial neuron computes a weighted sum and applies a threshold to make binary decisions.
- How the perceptron learning rule iteratively adjusts weights based on prediction errors.
- Why feature scaling matters for stable and faster convergence.
- The perceptron can only solve **linearly separable** problems (e.g., AND, OR, NAND, and this Pass/Fail dataset) — it cannot solve non-linearly separable problems like XOR, which is why multi-layer neural networks (MLPs) were later developed.

## 🛠️ Tech Stack

- Python 3
- NumPy

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
