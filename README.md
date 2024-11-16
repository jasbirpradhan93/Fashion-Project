This project implements a series of neural networks to classify images from the Fashion MNIST dataset, which contains 70,000 grayscale images of 10 clothing categories. The models range from simple fully connected networks to more advanced convolutional neural networks (CNNs). The project showcases model training, evaluation, and comparison.

Project Overview
The notebook explores the following topics:

Data preprocessing and normalization
Model building with TensorFlow and Keras (using different architectures)
Model evaluation using training and validation loss and accuracy
Comparison of multiple neural network architectures, including:
Simple fully connected model
Convolutional Neural Networks (CNNs) with increasing depth and complexity
CNN with dropout layers and batch normalization for improved accuracy
Visualization of performance metrics and misclassified samples
Plotting a confusion matrix and visual comparison of model accuracies
Dataset
The Fashion MNIST dataset contains 60,000 training images and 10,000 test images, each of size 28x28 pixels and labeled in one of 10 categories:

T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot
Requirements
Python 3.x
TensorFlow and Keras
Matplotlib
Numpy
Scikit-learn
Jupyter or Google Colab environment
Install the necessary libraries using:

bash
Copy code
pip install tensorflow numpy pandas matplotlib scikit-learn
Notebook Structure
Data Loading and Preprocessing

Load the Fashion MNIST dataset.
Normalize the pixel values to be between 0 and 1.
Split the training data into training and validation sets.
Model Architectures

Model #1: Fully connected neural network with two hidden layers.
Model #2: Shallow CNN with two convolutional layers and a dropout layer.
Model #3: Deeper CNN with three convolutional blocks.
Model #4: CNN with dropout, batch normalization, and a learning rate reduction callback.
Model Training

Compile each model with SparseCategoricalCrossentropy loss and the Adam optimizer.
Train each model and save training history for loss and accuracy metrics.
Model Evaluation

Evaluate each model on the test set.
Plot training and validation loss and accuracy for each model.
Display misclassified images.
Comparison of Model Performance

Plot accuracies of each model side-by-side.
Display a confusion matrix for the best-performing model.
Results
The models were trained for up to 50 epochs, with their performance compared based on test accuracy. Below are some notable findings:

Model #1 achieved a simple baseline accuracy.
Model #2 and Model #3 showed improved performance with deeper architectures.
Model #4 achieved the highest accuracy using dropout and batch normalization.
Running the Notebook
To run the notebook, open it in a Jupyter environment or Google Colab and execute each cell. The notebook will automatically download the Fashion MNIST dataset and proceed with training and evaluation.

Visualization
The notebook includes several visualizations:

Training and Validation Loss/Accuracy for each model.
Confusion Matrix to evaluate class-wise performance.
Misclassified Images for visual inspection of model predictions.
Accuracy Comparison Bar Plot showing test accuracy for each model.
License
This project is licensed under the MIT License.
