# Image-Classification-using-CNN
# Small Image Classification Using Convolutional Neural Network (CNN)
In this notebook, we will classify small images cifar10 dataset from tensorflow keras datasets. There are total 10 classes as shown below. We will use CNN for classification
The CIFAR-10 dataset consists of the following 10 classes:

0: Airplane
1: Automobile
2: Bird
3: Cat
4: Deer
5: Dog
6: Frog
7: Horse
8: Ship
9: Truck
# Prerequisites
Make sure you have Python 3.6 or higher installed. You will also need to install the required Python libraries.
# Required Libraries
TensorFlow (for ANN and CNN models)
NumPy (for numerical computations)
Matplotlib (for visualizations)
scikit-learn (for classification reports)
# Data Loading
The CIFAR-10 dataset is loaded using the tensorflow.keras.datasets.cifar10 module.
# Image Preprocessing
The images are normalized by dividing pixel values by 255 to scale them between 0 and 1. This helps in faster convergence during training.
# Model 1: Artificial Neural Network (ANN)
A simple fully connected ANN is built using the Keras Sequential API. The model consists of:

A flattening layer to reshape the input data
Two fully connected Dense layers with ReLU activations
An output layer with softmax activation for multi-class classification
The model is compiled with the Stochastic Gradient Descent (SGD) optimizer and trained on the training dataset for 5 epochs.

ANN Training Results:
Accuracy: 50.07%
Loss: 1.4271
# Model 2: Convolutional Neural Network (CNN)
A CNN model is built with the following layers:

Two convolutional layers with ReLU activation and max-pooling layers
A fully connected dense layer with ReLU activation
An output layer with softmax activation
The model is compiled using the Adam optimizer and trained for 10 epochs.

CNN Training Results:
Accuracy: 78.82%
Loss: 0.6076
# Conclusion
ANN performed with a lower accuracy (~50%) but showed significant potential.
CNN achieved better accuracy (~78%) and should be considered for more complex image classification tasks.
