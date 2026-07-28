# Real-Time Feedforward Neural Network Classifier

A simple PyTorch project that trains a small feedforward neural network once, then simulates a live stream of incoming data points — classifying each one the moment it "arrives" and plotting it in real time.

## 📌 Overview

This is a simplified, beginner-friendly version of a real-time streaming classifier. It demonstrates:

1. **Offline training** — a small neural network learns to separate two classes from a labeled dataset.
2. **Real-time simulation** — new, unseen points arrive one at a time (like sensor readings or transactions), and the trained model classifies each one instantly.
3. **Live visualization** — a single plot updates point by point, colored by predicted class, along with a running accuracy printed to the console.

## 🧠 Model Architecture

A small feedforward neural network with one hidden layer:

```
Input (2 features) → Linear(2, 8) → ReLU → Linear(8, 1) → Sigmoid → Output (0 or 1)
```

- **ReLU** activation in the hidden layer introduces non-linearity.
- **Sigmoid** activation in the output layer squashes the result into a probability between 0 and 1.
- A prediction ≥ 0.5 is classified as class 1, otherwise class 0.

## 📂 Files

| File | Description |
|------|-------------|
| `realtime_ffnn_simple.py` | Main script — trains the model and simulates the real-time stream |
| `realtime_result.png` | Auto-generated snapshot of the final plot (created after running the script) |

## ⚙️ How It Works

### 1. Training (Offline)

The model is trained once on a labeled 2D dataset generated using `sklearn.datasets.make_classification`, using the Adam optimizer and Binary Cross-Entropy loss for 150 epochs.

### 2. Streaming Simulation

A separate, unseen batch of 30 points is generated to simulate a live stream. The script loops through them one at a time:
- Feeds the point into the trained model
- Gets a prediction and confidence score
- Plots the point immediately, colored by predicted class (red = class 1, blue = class 0)
- Prints the prediction, confidence, actual label, and running accuracy to the console

### 3. Output

- A live matplotlib window shows points appearing one by one.
- A final snapshot is saved as `realtime_result.png`.

## ▶️ How to Run

1. Install dependencies:
   ```bash
   pip install numpy torch matplotlib scikit-learn
   ```
2. Run the script:
   ```bash
   python realtime_ffnn_simple.py
   ```
3. Watch the plot update live as each point is classified. Console output will show predictions and running accuracy as they happen.

## 📈 Sample Console Output

```
Training done. Final loss: 0.1234
Point 1/30 | Predicted: 0 (confidence 0.12) | Actual: 0 | Running accuracy: 100.0%
Point 2/30 | Predicted: 1 (confidence 0.87) | Actual: 1 | Running accuracy: 100.0%
...
Final snapshot saved as realtime_result.png
```

## 🎯 Key Learnings

- How a simple feedforward network with ReLU and Sigmoid activations performs binary classification.
- The difference between **offline training** (learning from a fixed dataset) and **real-time inference** (making predictions on new data as it arrives).
- How to visualize model predictions incrementally rather than all at once.

## 🛠️ Tech Stack

- Python 3
- PyTorch
- scikit-learn
- Matplotlib

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
