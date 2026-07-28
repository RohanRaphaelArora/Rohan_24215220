# Real-Time Feedforward Neural Network Classifier

A simple PyTorch project that trains a Feedforward Neural Network (FFNN) for binary classification and demonstrates real-time prediction by classifying incoming data points one at a time. The visualization updates continuously while displaying prediction confidence and running accuracy.

---

##  Overview

This project demonstrates the complete workflow of a basic Feedforward Neural Network:

1. Generate a synthetic binary classification dataset.
2. Train a neural network using the Adam optimizer.
3. Simulate a real-time stream of unseen data.
4. Predict one incoming sample at a time.
5. Display predictions graphically while tracking running accuracy.

---

##  Model Architecture

The neural network consists of:

Input Layer (2 Features)
        ↓
Linear Layer (2 → 8)
        ↓
ReLU Activation
        ↓
Linear Layer (8 → 1)
        ↓
Sigmoid Activation
        ↓
Binary Output (0 or 1)

---

##  Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.05 |
| Loss Function | Binary Cross Entropy Loss (BCELoss) |
| Epochs | **150** |
| Training Samples | 200 |
| Streaming Samples | 30 |
| Output Classes | 2 (Binary Classification) |

---

## 📂 Files

| File | Description |
|------|-------------|
| realtime_ffnn_simple.py | Main implementation of the Feedforward Neural Network |
| realtime_result.png | Final snapshot generated after the streaming simulation |

---

## 🚀 How It Works

### 1. Training Phase

- A synthetic dataset containing 200 samples is generated using `make_classification()`.
- The Feedforward Neural Network is trained for **150 epochs**.
- The Adam optimizer minimizes the Binary Cross Entropy Loss until convergence.

### 2. Real-Time Prediction

A second unseen dataset is generated to simulate incoming live data.

For every incoming sample:

- The model predicts its class.
- Prediction confidence is computed.
- Running accuracy is updated.
- The scatter plot refreshes in real time.
- Predicted points are colored according to their predicted class.

---

## ▶️ How to Run

### Install dependencies

```bash
pip install torch numpy matplotlib scikit-learn
```

### Run the program

```bash
python realtime_ffnn_simple.py
```

The console will display the prediction, confidence score, actual class, and running accuracy for every incoming data point. A final visualization is saved as `realtime_result.png`.

---

## 📊 Observations

- The Feedforward Neural Network successfully learned the decision boundary after training for **150 epochs**.
- The Binary Cross Entropy Loss decreased steadily, indicating successful model convergence.
- The model classified unseen streaming samples individually without requiring retraining.
- Prediction confidence varied depending on how close each sample was to the learned decision boundary.
- Running accuracy remained relatively stable throughout the streaming process, demonstrating good generalization on unseen data.
- The real-time visualization clearly illustrated how new predictions accumulated over time.
- The project demonstrates how Feedforward Neural Networks can be adapted for online-style inference, where incoming samples are processed one at a time.

---

## 📈 Results

- Successfully trained a binary classifier using a Feedforward Neural Network.
- Performed real-time prediction on 30 unseen samples.
- Displayed prediction confidence for every incoming sample.
- Continuously updated running classification accuracy.
- Generated a graphical visualization of all classified samples.

---

## 💡 Key Learnings

- Understanding the architecture of a Feedforward Neural Network.
- Binary classification using PyTorch.
- Training neural networks using Adam Optimization.
- Binary Cross Entropy Loss for classification.
- Real-time inference on streaming data.
- Visualization of sequential predictions using Matplotlib.
- Performance monitoring using running accuracy.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy
- Scikit-learn
- Matplotlib

---

## 👨‍💻 Author

**Rohan Raphael Arora**

Data Science & Artificial Intelligence
