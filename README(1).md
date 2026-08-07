# Convolutional Neural Network (CNN) - Cats vs Dogs Classifier

## Lab #5: Using TensorFlow/Keras and the Cats vs Dogs Dataset to Perform a CNN Model

This lab builds a Convolutional Neural Network (CNN) using **TensorFlow/Keras** that
classifies images into two classes — **Cat** or **Dog**. It is fully application
oriented: run the notebook top to bottom and it downloads its own dataset,
trains the model, and predicts on a real image at the end.

---

### Importing the libraries

We use `tensorflow` and Keras's `ImageDataGenerator` for loading and augmenting the
image dataset (rescaling, shearing, zooming, flipping).

---

## Part 0 - Downloading the Dataset

Instead of manually uploading images or mounting Google Drive, the notebook
automatically downloads a small, ready-to-use **Cats vs Dogs** dataset
(~2,000 training images + 1,000 validation images) directly from Google's
public storage using `tf.keras.utils.get_file()`. This makes the notebook
runnable anywhere — Colab, Jupyter, or your local machine — with zero setup.

---

## Part 1 - Data Preprocessing

### Preprocessing the Training set
Training images are rescaled (pixel values 0–1) and augmented with random shear,
zoom, and horizontal flips using `ImageDataGenerator`, then loaded in batches of
32 with `flow_from_directory`.

### Preprocessing the Test set
Test/validation images are only rescaled (no augmentation), since we want to
evaluate the model on unmodified images.

---

## Part 2 - Building the CNN

### Initialising the CNN
A `Sequential` Keras model is created as the container for our CNN layers.

### Step 1 - Convolution
A `Conv2D` layer with 32 filters (3x3 kernel) and ReLU activation extracts
low-level features (edges, textures) from the 64x64x3 input images.

### Step 2 - Pooling
A `MaxPool2D` layer (2x2, stride 2) down-samples the feature maps, keeping the
strongest activations and reducing computation.

### Adding a second convolutional layer
A second `Conv2D` + `MaxPool2D` pair is added so the network can learn more
complex, higher-level features from the pooled output of the first layer.

### Step 3 - Flattening
The 2D feature maps are flattened into a single 1D vector so they can be fed
into the fully connected (dense) layers.

### Step 4 - Full Connection
A `Dense` layer with 128 neurons and ReLU activation learns to combine the
extracted features to distinguish cats from dogs.

### Step 5 - Output Layer
A single `Dense` neuron with **sigmoid** activation outputs a probability
between 0 and 1 — this is a binary classification problem (0 = cat, 1 = dog).

---

## Part 3 - Training the CNN

### Compiling the CNN
The model is compiled with:
- **Optimizer:** `adam` — how the model updates its weights.
- **Loss Function:** `binary_crossentropy` — how the model measures prediction error for two-class problems.
- **Metrics:** `accuracy` — how the model's performance is tracked during training.

### Training the CNN on the Training set and evaluating it on the Test set
The model is trained (`cnn.fit`) on the training set for several epochs, with
the test/validation set used to check performance on unseen images after every
epoch.

---

## Part 4 - Making a Single Prediction

This is the application-oriented step: a real image is loaded, resized to
64x64, converted to an array, and passed through the trained CNN. Based on
`training_set.class_indices`, the output is mapped back to a human-readable
label — **"cat"** or **"dog"**.

### (Optional) Try your own image
You can swap in your own uploaded photo (e.g., in Colab, use the file upload
button) and change the image path in the last cell to see the model predict
on it directly.

---

## How to Run
1. Open `Lab5_CNN_Cats_vs_Dogs.ipynb` in Google Colab (recommended, free GPU)
   or Jupyter Notebook.
2. Run all cells in order, top to bottom.
3. The dataset downloads automatically — no manual steps needed.
4. Training takes a few minutes on a GPU runtime (Colab: **Runtime > Change
   runtime type > GPU**).
5. The final cells show the model's prediction on a sample image.

## Requirements
- Python 3.x
- TensorFlow 2.x (`pip install tensorflow`)
- NumPy (`pip install numpy`)

## Dataset
[Cats and Dogs Filtered](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip)
— a small filtered subset of the Kaggle "Dogs vs. Cats" dataset, hosted by
Google for educational use.
