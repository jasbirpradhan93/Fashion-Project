# Fashion MNIST Classification Project

This repository implements various neural networks to classify images from the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), which contains 70,000 grayscale images across 10 categories of clothing. The project includes model training, evaluation, and comparison, highlighting the evolution from simple neural networks to more advanced convolutional architectures.

## Project Overview

This project includes the following components:
- **Data Preprocessing**: Normalization and train-validation splitting
- **Model Building**: Implementing models with increasing complexity
- **Model Evaluation**: Using accuracy, loss metrics, and confusion matrices
- **Comparison**: Evaluating multiple architectures and visualizing their performance

## Dataset

The Fashion MNIST dataset includes:
- **60,000 training images** and **10,000 test images**.
- Each image is 28x28 pixels and belongs to one of the following categories:
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot

## Requirements

The following libraries are required to run the code:
- `Python 3.x`
- `TensorFlow` and `Keras`
- `Matplotlib`
- `NumPy`
- `Scikit-learn`

Install dependencies with:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn


# Model Architectures

Model #1: Fully connected neural network with two hidden layers.
Model #2: Shallow CNN with two convolutional layers and a dropout layer.
Model #3: Deeper CNN with three convolutional blocks.
Model #4: CNN with dropout, batch normalization, and a learning rate reduction callback.

# Model Training
Compile each model with SparseCategoricalCrossentropy loss and the Adam optimizer.
Train each model and save training history for loss and accuracy metrics.

# Model Evaluation
Evaluate each model on the test set.
Plot training and validation loss and accuracy for each model.
Display misclassified images.
Comparison of Model Performance
Plot accuracies of each model side-by-side.
Display a confusion matrix for the best-performing model.

# Results
The models were trained for up to 50 epochs, with their performance compared based on test accuracy. Below are some notable findings:

Model #1 achieved a simple baseline accuracy.
Model #2 and Model #3 showed improved performance with deeper architectures.
Model #4 achieved the highest accuracy using dropout and batch normalization.

# Running the Notebook
To run the notebook, open it in a Jupyter environment or Google Colab and execute each cell. The notebook will automatically download the Fashion MNIST dataset and proceed with training and evaluation.

# Visualization
The notebook includes several visualizations:
Training and Validation Loss/Accuracy for each model.
Confusion Matrix to evaluate class-wise performance.
Misclassified Images for visual inspection of model predictions.
Accuracy Comparison Bar Plot showing test accuracy for each model.

# License
This project is licensed under the MIT License.
