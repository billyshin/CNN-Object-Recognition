"""
Convolutional nerual network (CNN) for object recognition.
"""
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout, Activation, Conv2D, GlobalAveragePooling2D
from keras.optimizers import SGD
from matplotlib import pyplot as plt 
import numpy as np
from PIL import Image

# ====================================== Loading Data ======================================
"""
The dataset we will be using is the CIFAR-10 dataset, which consists of 60,000 32x32 color images 
in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
"""
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Check the data
print('Training Images: {}'.format(X_train.shape))  # Training Images: (50000L, 3L, 32L, 32L)
print('Testing Images: {}'.format(X_test.shape))  # Testing Images: (10000L, 3L, 32L, 32L)

# Single image
print(X_train[0].shape)  # (3L, 32L, 32L)

# Create a grid of 3x3 images
for i in range(9):
    plt.subplot(300 + 1 + i)
    img = X_train[i].transpose([1, 2, 0])
    plt.imshow(img)

# Plot
plt.show()


# ====================================== Preprocessing the dataset ======================================
"""
Building a convolutional neural network for object recognition on CIFAR-10
"""
np.random.seed(6)

# Normalize the inputs from 0 - 255 to 0.0 - 1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# class labels shape
print(y_train.shape)  # (50000L, 1L)
print(y_train[0])  # [6]

# hot encode outputs
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)
num_classes = Y_test.shape[1]

# check outputs
print(Y_train.shape)  #(50000L, 10L)
print(Y_train[0])  #[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]


# ====================================== Building the All-CNN ======================================
"""
The architecture of the CNN is stated  in the 2015 ICLR paper, "Striving For Simplicity: The All Convolutional Net".
"""
def allcnn(weights=None):
    # define model type - Sequential
    model = Sequential()

    # add model layers - Convolution2D, Activation, Dropout
    model.add(Conv2D(96, (3, 3), padding = 'same', input_shape=(3, 32, 32)))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding = 'same', strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding = 'same', strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1), padding = 'valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1), padding = 'valid'))

    # add GlobalAveragePooling2D layer with Softmax activation
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    
    # load the weights
    if weights:
        model.load_weights(weights)
    
    # return model
    return model


# ====================================== Defining Parameters and Training the Model ======================================
# define hyper parameters
learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9

# build model 
model = allcnn()

# define optimizer and compile model
sgd = SGD(lr=learning_rate, decay=weight_decay, momentum=momentum, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# print model summary
print (model.summary())

# define additional training parameters
epochs = 350
batch_size = 32

# fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size, verbose = 1)


# ====================================== Using GPU to Train the Deep Convolutional Neural Network ======================================
# define hyper parameters
learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9

# define weights and build model
model = allcnn('all_cnn_weights_0.9088_0.4994.hdf5')

# define optimizer and compile model
sgd = SGD(lr=learning_rate, decay=weight_decay, momentum=momentum, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# print model summary
print (model.summary())

# test the model with pretrained weights
scores = model.evaluate(X_test, Y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


# ====================================== Making Prediction ======================================
# make dictionary of class labels and names
classes = range(0,10)

names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# zip the names and classes to make a dictionary of class_labels
class_labels = dict(zip(classes, names))

# generate batch of 9 images to predict
batch = X_test[100:109]
labels = np.argmax(Y_test[100:109],axis=-1)

# make predictions
predictions = model.predict(batch, verbose = 1)

# print our predictions
print(predictions)

# these are individual class probabilities, should sum to 1.0 (100%)
for image in predictions:
    print(np.sum(image))

# use np.argmax() to convert class probabilities to class labels
class_result = np.argmax(predictions,axis=-1)
print(class_result)

# create a grid of 3x3 images
fig, axs = plt.subplots(3, 3, figsize = (15, 6))
fig.subplots_adjust(hspace = 1)
axs = axs.flatten()

for i, img in enumerate(batch):
    # determine label for each prediction, set title
    for key, value in class_labels.items():
        if class_result[i] == key:
            title = 'Prediction: {}\nActual: {}'.format(class_labels[key], class_labels[labels[i]])
            axs[i].set_title(title)
            axs[i].axes.get_xaxis().set_visible(False)
            axs[i].axes.get_yaxis().set_visible(False)
            
    # plot the image
    axs[i].imshow(img.transpose([1,2,0]))
    
# show the plot
plt.show()
