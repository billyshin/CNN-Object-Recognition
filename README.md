# CNN-Object-Recognition

A Convolutional Neural Network (CNN) for object recognition.

## Step 1: Necesaary Packages:
  - tensorflow
  - keras
  - matplotlib
  - numpy
  - PIL
  
## Step 2: Loading the Data:
The dataset we will be using is the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6000 images per class. There are 50,000 training images and 10,000 testing images.
CIFAR-10 dataset can be found in https://www.cs.toronto.edu/~kriz/cifar.html

## Step 3: Preprocessing the Dataset:
We need to preprocess the dataset so the images and labels are in a form that Keras can ingest. First of all, we will normalize the images. Furthermore, we will also convert our class label, a single integer value (0-9) to one-hot vector of length of ten, e.g. the class label of 6 should be denoted  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]. This is a standard output format for neural networks. 

## Step 4: Building the All-CNN:
Using the 2015 ICLR paper, "Striving For Simplicity: The All Convolutional Net" as a reference, we can implement the All-CNN network in Keras.

## Step 5: Defining Parameters and Training the Model:
Define hyper parameters, such as learning rate and momentum, define an optimizer, compile the model, and fit the model to the training data. (See the details in code)

## Step 6: Making Predictions:
Leverage the network to make prediction based on trained weights. To start, we will generate a dictionary of class labels. Next, we will make predictions on nine images and compare the results to the ground-truth labels. Moreover, we will plot the images for visual reference.
Sample result:

![alt text](https://github.com/billyshin/CNN-Object-Recognition/blob/master/Screen-Shot-2018-12-25-at-5.34.50-AM.png)
