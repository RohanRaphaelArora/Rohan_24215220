# Lab 8: Implementation of RNN-Based Text Classification

**Application:** Movie Review Sentiment Analyzer
**Course:** Artificial Neural Networks and Deep Learning
**Dataset:** IMDB Movie Reviews (`tensorflow.keras.datasets`)
**Framework:** TensorFlow / Keras

---

## 1. Aim

To implement a Recurrent Neural Network (SimpleRNN) in TensorFlow/Keras that classifies movie reviews as **Positive (1)** or **Negative (0)**, and to evaluate its performance on the IMDB dataset.

## 2. Application Overview

The project works as a small **sentiment analyzer**: you type any movie review in English and the trained RNN tells you whether it is Positive or Negative, along with its confidence. Everything before that (loading data, training, graphs, experiments) is there to build and understand the model behind it.

## 3. Requirements

```
pip install tensorflow matplotlib pandas numpy
```

Python 3.9 or newer. Works on Google Colab without any setup.

## 4. How to Run

1. Open `rnn_sentiment_analyzer.py` in Google Colab / Jupyter (each `# %%` marker is a cell), **or** run `python rnn_sentiment_analyzer.py`.
2. Run all cells from top to bottom. The experiments in Task 7 take a few minutes.
3. At the end, type your own reviews into the live analyzer. Type `quit` to stop.

**Files produced:** `accuracy_graph.png`, `loss_graph.png`, `experiment_results.csv`

## 5. Code Structure

| Section in code | What it does |
|---|---|
| Imports and Settings | Imports libraries; sets vocabulary = 10,000, max length = 200, embedding = 128 |
| Task 1 | Loads IMDB and prints sample counts, shapes, positive/negative counts |
| Task 2 | Pads/truncates every review to 200 words and prints final shapes |
| Task 3 | `build_model()` builds Embedding → SimpleRNN → Dense(sigmoid) and prints the summary |
| Task 4 | Trains with Adam, binary cross-entropy, accuracy, batch 64, validation split 20% |
| Task 5 | Evaluates on test data and answers Q1 and Q2 |
| Task 6 | Plots the accuracy graph and loss graph |
| Task 7 | Runs experiments A (units), B (epochs), C (batch size) and builds the results table |
| Task 8 | `predict_sentiment()` predicts five new reviews |
| Task 9 | Compares single-layer and two-layer RNN (`return_sequences=True`) |
| Discussion Questions | Answers to all 10 questions |
| Conclusion | Summary filled in from your actual results |
| Live Application | Interactive analyzer for your own reviews |

## 6. Task 1: Load and Explore the Dataset

- Vocabulary size: **10,000**; maximum sequence length: **200**
- Prints training/testing sample counts, data shapes, and positive/negative counts (25,000 train / 25,000 test, evenly balanced).

## 7. Task 2: Data Preprocessing

- The vocabulary is limited with `num_words=10000`.
- Reviews are already integer sequences; `pad_sequences(maxlen=200)` makes all reviews the same length.
- Final shapes: training `(25000, 200)`, testing `(25000, 200)`.

## 8. Task 3: Build the RNN Classifier

```
Input (200) → Embedding (10000 → 128) → SimpleRNN (64) → Dense (1, sigmoid)
```

The model summary is printed by the code.

## 9. Task 4: Compile and Train the Model

- Optimizer: **Adam**; Loss: **Binary Cross-Entropy**; Metric: **Accuracy**
- Epochs: 5, Batch size: 64, Validation split: 20%
- Training/validation accuracy and loss are printed after training.

## 10. Task 5: Evaluate the Model

Test loss and test accuracy are printed. The code also answers:

- **Q1:** the final test accuracy (printed as a percentage).
- **Q2:** whether test accuracy is higher or lower than training accuracy and why (usually lower, due to overfitting).

## 11. Task 6: Performance Visualization

- **Graph 1:** Training vs validation accuracy per epoch
- **Graph 2:** Training vs validation loss per epoch

Interpretation: if training accuracy keeps rising but validation accuracy flattens or drops (or validation loss starts rising), the model is overfitting.

## 12. Task 7: Hyperparameter Experimentation

| Experiment | Parameter | Training Accuracy | Validation Accuracy | Test Accuracy |
|---|---|---|---|---|
| A1 | RNN = 32 | | | |
| A2 | RNN = 64 | | | |
| A3 | RNN = 128 | | | |
| B1 | Epochs = 3 | | | |
| B2 | Epochs = 5 | | | |
| B3 | Epochs = 10 | | | |
| C1 | Batch = 32 | | | |
| C2 | Batch = 64 | | | |
| C3 | Batch = 128 | | | |

*(Copy your values from the printed table / `experiment_results.csv`. A2, B2 and C2 are the same configuration, so the code trains it only once.)*

## 13. Task 8: Text Prediction

`predict_sentiment(review)` converts text to the IMDB word indices, pads it to 200, and returns the predicted label with the probability of the review being positive.

| Review | Expected Sentiment | Predicted Sentiment | Probability |
|---|---|---|---|
| Review 1 | Positive | | |
| Review 2 | Negative | | |
| Review 3 | Positive | | |
| Review 4 | Negative | | |
| Review 5 | Positive | | |

## 14. Task 9: Challenge Activity

Model: `Embedding → SimpleRNN(return_sequences=True) → SimpleRNN → Dense`

1. **Better test accuracy?** Printed by the code (compare the two models).
2. **Longer to train?** Usually the two-layer model, since it has more computation per step.
3. **Does another layer always help?** No. It adds parameters and can worsen vanishing gradients and overfitting.
4. **Purpose of `return_sequences=True`:** it makes the layer return the output at every time step, not only the last one, which the next RNN layer needs as input.

## 15. Discussion Questions

1. **What is an RNN and why is it suitable for text?** An RNN processes data step by step and carries a hidden state forward, so it remembers earlier words. Word order matters in text.
2. **Purpose of the Embedding layer?** Converts word indices into dense learned vectors where similar words are close together.
3. **Why is padding required?** Batches need a fixed size, so all reviews are made 200 words long.
4. **Why sigmoid?** It outputs a value between 0 and 1, interpreted as the probability of a positive review.
5. **Why binary cross-entropy?** It is the standard loss for two-class problems and penalises confident wrong predictions strongly.
6. **Vanishing-gradient problem?** Gradients shrink as they travel back through many time steps, so early words barely influence learning.
7. **More RNN units?** More learning capacity, but slower training and a higher risk of overfitting.
8. **What is overfitting?** Great performance on training data, poor performance on unseen data.
9. **How to reduce overfitting?** Dropout, early stopping, fewer units, more data, regularisation.
10. **SimpleRNN vs LSTM vs GRU?** SimpleRNN has a single simple hidden state; LSTM adds a memory cell with three gates for long-term memory; GRU uses two gates and is lighter and faster with similar accuracy.

## 16. Conclusion

The code prints a conclusion (about 150-200 words) using your actual results. It covers what was learned, how the RNN performed, the best hyperparameters, the effect of units and epochs, limitations (vanishing gradients, overfitting), and how LSTM/GRU could improve the system. Rewrite it in your own words before submitting.

## 17. Submission Checklist

- [ ] Notebook (`.ipynb`) run from start to end with all outputs visible
- [ ] Lab report in PDF (graphs, results table, answers, conclusion)
- [ ] Graphs have titles and axis labels (already done in the code)
- [ ] Comments in code are understood and explained in your own words
