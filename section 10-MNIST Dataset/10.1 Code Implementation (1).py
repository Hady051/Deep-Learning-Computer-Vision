#### Importing the Libraries
##"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist   # // loading the mnist dataset is from keras not sklearn
from keras.models import Sequential
from keras.layers import Dense      # // DNN
from keras.optimizers import Adam
from keras.utils import to_categorical # // multi-classification (One Hot Encoding)

import random

# // he didn't use this
# from sklearn.model_selection import train_test_split

#@"""



#### Importing the Data/ Train Test Split
##"""

np.random.seed(0)

# // here, we won't use the train_test_split function: (X_train, X_test, y_train, y_test)
# // instead:
# (X_train, y_train), (X_test, y_test), X:input features (MNIST images), 28x28 pixels

(images_train, labels_train), (images_test, labels_test) = mnist.load_data()

#// this returns 60,000 MNIST images to the train set and 10,000 ones to the test set

print("images_train.shape = ", images_train.shape)  # // images_train.shape =  (60000, 28, 28)
print("images_test.shape = ", images_test.shape)    # // images_test.shape =  (10000, 28, 28)
print("The number of labels: ", labels_train.shape[0] )
# // this shows 60,000 which is 60,000 labels for the 60,000 images

#@"""



#### assert Method (ensure the imported data is accurate)
##"""

# // This is useful whenever you want to import complex datasets.
# // The (assert) method takes in 1 arg,
# // the arg is usually just a condition that is either going to be True or False
# // If the condition is met, the code will run smoothly with no issues,
# // However, if the condition is not met, the code will stop running and will display an error to alert
# // the user that something went wrong.

# // this confirms that 60k images have 60k labels, if not, the code won't run
assert(images_train.shape[0] == labels_train.shape[0] ), "The number of images isn't equal to the number of labels"
# // same for the 10k of the test set
assert(images_test.shape[0] == labels_test.shape[0] ), "The number of images isn't equal to the number of labels"
# // the same goes for the shape of the images
assert (images_train.shape[1: ] == (28, 28) ), "The dimensions of the images are not 28x28."
assert (images_test.shape[1: ] == (28, 28) ), "The dimensions of the images are not 28x28."

#@"""



####
##"""

number_of_samples = []
number_of_classes = 10
cols = 5

#@"""



#### plotting samples of the MNIST Dataset
##"""

fig, axes = plt.subplots(nrows=number_of_classes, ncols=cols, figsize=(5, 8) )
# // figsize is the size of each grid (subplot), first the width then the height
# // You can see the effect of figsize when you minimize the pic, not maximize it.
fig.tight_layout()
# // this is used to automatically deal with overlapping and separates the figures/subplots better

for col in range(cols): # // each col of the 5
    for row in range(number_of_classes): # // each row(class) of the 10
        image_selected = images_train[labels_train == row] # // this is used to split the data,
        # // with the conditional statement [labels_train == col(name of class)],
        # // so that row 0 will have MNIST images of class 0, etc…
        axes[row][col].imshow(image_selected[random.randint(0, len(image_selected) - 1), : , : ],
                          cmap=plt.get_cmap("gray") )
        # // random.randint(), gets you a random integer between the 2 integers you entered.
        # // leave the rest blank (:), to ensure you get the full image, the full (28x28) image
        # // the images get displayed but not properly, and that's because
        # // the default colormap of matplotlib is (Viridis),
        # // which is a colorful color map that is useful for some data.
        # // in our case, we want a grayscale images.

        # // if you don't want the images to be in a labeled axis
        axes[row][col].axis("off")

        if col == 2:
            axes[row][col].set_title("Class: " + str(row) ) # // to name the class with the value it has

            number_of_samples.append(len(image_selected) )
        # // You need to include this inside the if block, to take only one sample of the 10 classes.
        # // which is in col = 2 in this case. You can do another if with whatever col you want,it's the same
        # // if you don't, 5 samples of the 10 classes (bec of no. of cols) will be added which is WRONG.

#@"""



#### Bar plot to show the number of images each class has
"""

print(number_of_samples)
# // this shows the amount of images belonging to each class.
# // [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]

# // To visualize this with a bar-plot

plt.figure(figsize=(12, 4) )
plt.bar(range(0, number_of_classes), number_of_samples, color="turquoise", width=0.5)
# // first arg is the x-coordinate (each class),
# // 2nd arg is the y-coordinate (number of images per each class)-(number_of_samples array)
# // width arg is the width of the bars.
plt.title('Distribution of the training dataset')
plt.xlabel('Class')
plt.ylabel('Number of images')

plt.show()

"""



#### One Hot Encoding
##"""

labels_train_OHE = to_categorical(labels_train, 10)
labels_test_OHE = to_categorical(labels_test, 10)

#@"""



#### Normalization (Feature Scaling)
##"""

# // here we just divide by 255
images_train = images_train / 255
images_test = images_test / 255

# // well, for instance, our training data are 60k images.
# // Each image has a pixel intensity ranging between 0 and 255.
# // So now we want to scale it between 0 and 1
# // This normalization process is important as it scales down our features to a uniform range and decreases
# // variance among our data.
# // Due to the nature of the mathematical operations used inside the neural network.
# // We need to ensure that our data has low variance.
# // This helps the neural network better deal with the input data and to learn more quickly and accurately.

#@"""



#### Flattening / Reshaping
##"""

print("images_train shape = ", images_train.shape) # // images_train shape = (60000, 28, 28)
print("images_test shape = ", images_test.shape)  # // images_test shape = (10000, 28, 28)

# // Now that we have a normalized data set, we are going to flatten our images.
# // We currently have images that have the shape of a two-dimensional array.
# // Each image is 28 by 28 pixels, 784 pixels in total.
# // However, due to the structure of our neural network, our input values are going to be multiplied by
# //the weight matrix connecting our input layer to our first hidden layer.
# // To conduct matrix multiplication, we must make our images one dimensional.
# // We cannot pass the array in the way that it's currently shaped.
# // That is, instead of them being 28 rows by 28 columns, we must flatten each image into a single row of 784 pixels.

number_of_pixels = 28 * 28  # // 784
images_train = images_train.reshape(images_train.shape[0], number_of_pixels )
images_test = images_test.reshape(images_test.shape[0], number_of_pixels)
# // we won't change the first dim (60k), we will change the 28 x 28 to 1D of 784 (flatten it to 1D)

print("images_train new shape = ", images_train.shape) # // images_train shape = (60000, 784)
print("images_test new shape = ", images_test.shape)  # // images_test shape = (10000, 784)

#@"""
