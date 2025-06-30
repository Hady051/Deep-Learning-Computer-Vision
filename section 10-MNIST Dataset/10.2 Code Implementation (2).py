###### Part 1
# ** press on a variable, shift + F6 twice to change a variable name throughout all the file.


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

#@"""



#### Importing the Data/ Train Test Split
##"""

np.random.seed(0)

(images_train, labels_train), (images_test, labels_test) = mnist.load_data()

print("images_train.shape = ", images_train.shape)  # // images_train.shape =  (60000, 28, 28)
print("images_test.shape = ", images_test.shape)    # // images_test.shape =  (10000, 28, 28)
print("The number of labels: ", labels_train.shape[0] )
# // this shows 60,000 which is 60,000 labels for the 60,000 images

#@"""



#### assert Method (ensure the imported data is accurate)
##"""

# // this confirms that 60k images have 60k labels, if not, the code won't run
assert(images_train.shape[0] == labels_train.shape[0] ), "The number of images isn't equal to the number of labels"

assert(images_test.shape[0] == labels_test.shape[0] ), "The number of images isn't equal to the number of labels"

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
"""

fig, axes = plt.subplots(nrows=number_of_classes, ncols=cols, figsize=(5, 8) )

fig.tight_layout()

for col in range(cols): # // each col of the 5
    for row in range(number_of_classes): # // each row(class) of the 10
        image_selected = images_train[labels_train == row]

        axes[row][col].imshow(image_selected[random.randint(0, len(image_selected) - 1), : , : ],
                          cmap=plt.get_cmap("gray") )

        axes[row][col].axis("off")

        if col == 2:
            axes[row][col].set_title("Class: " + str(row) )

            number_of_samples.append(len(image_selected) )

print(number_of_samples)
"""



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
images_train_N = images_train / 255
images_test_N = images_test / 255

#@"""



#### Flattening / Reshaping
##"""

print("images_train shape = ", images_train_N.shape) # // images_train shape = (60000, 28, 28)
print("images_test shape = ", images_test_N.shape)  # // images_test shape = (10000, 28, 28)

number_of_pixels = 28 * 28  # // 784
images_train_N = images_train_N.reshape(images_train_N.shape[0], number_of_pixels )
images_test_N = images_test_N.reshape(images_test_N.shape[0], number_of_pixels)
# // we won't change the first dim (60k), we will change the 28 x 28 to 1D of 784 (flatten it to 1D)

print("images_train new shape = ", images_train_N.shape) # // images_train shape = (60000, 784)
print("images_test new shape = ", images_test_N.shape)  # // images_test shape = (10000, 784)

#@"""



####################################################################################################
###### Part 2



#### Creating the model
##"""

# // It's better to use CNN for images than DNN. For this section, we will use DNN
# // listen to the video when writing.

def create_NN_model(num_of_pixels_of_pictures):
    nn_model = Sequential()


    # // We won't use the sigmoid activation function
    # // Instead, we will use the "relu" activation function. Will be discussed later.
    # // here, I used input_dim instead of input_shape. It's the same
    # // the input nodes is going to be the total number of pixels. (28x28)
    # // relu is similar to sigmoid and tanh, which is a non-linear function used to
    # // convert scores into probabilities but generally performs better.

    ## 1st hidden layer
    nn_model.add(Dense(input_dim=num_of_pixels_of_pictures, units=10, activation="relu") )
    # nn_model.add(Dense(input_dim=num_of_pixels_of_pictures, units=784, activation="relu") )
    ## 2nd hidden layer
    nn_model.add(Dense(units=10, activation="relu") )
    ## output layer
    nn_model.add(Dense(units=number_of_classes, activation="softmax") )  # // used in output layer for multi-class

    nn_model.compile(optimizer=Adam(learning_rate=0.01), loss="categorical_crossentropy", metrics=['accuracy'] )

    return nn_model

# // ** In the older file I made the first layer have units=number_of_pixels (784) instead of 10
# // this increased the capacity(complexity) of the model, and greatly increased the accuracy
# // from 92-94 % to 99.8 %

#@"""



''' example of other way to make NN
Model = Sequential()
inputs = Input(shape=(784,))                 # input layer
x = Dense(32, activation='relu')(inputs)     # hidden layer
outputs = Dense(10, activation='softmax')(x) # output layer

model = Model(inputs, outputs)
'''



#### NN model summary
"""

nn_model = create_NN_model(num_of_pixels_of_pictures=number_of_pixels)

# // first model summary
print(nn_model.summary() )
'''
┌─────────────────────────────────┬───────────────────────────┬────────────┐
│ Layer (type)                    │ Output Shape              │    Param # │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense (Dense)                   │ (None, 10)                │      7,850 │  # // (784 x 10) + 10 (bias values)
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_1 (Dense)                 │ (None, 10)                │        110 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_2 (Dense)                 │ (None, 10)                │        110 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 8,070 (31.52 KB)         
 Trainable params: 8,070 (31.52 KB)
 Non-trainable params: 0 (0.00 B)
'''

# // parameters are wights and biases.
# // for the parameters between input layer and 1st hidden layer: (784 x 10) + 10 (bias values for each node)
# // no. of input nodes x no. of hidden layer nodes + biases.

"""



#### Fitting the Model
"""

nn_model_fit = nn_model.fit(images_train_N, labels_train_OHE, validation_split=0.1, batch_size=200, epochs=12,
             verbose=True, shuffle=True)

# // "validation_split" arg is for the validation set, we left 10% of the training data for it.
# // This is for generalization. Check 10.3 and 10.4 sections
# // verbose and shuffle = 1 or (True)

"""



#### saving/loading the model
##"""

from keras.models import load_model

# // creates a HDF5 file nn_model.save('my_model.h5')  # / the hdf5 is wrong(old way)

# / creates a keras file (new way of saving)
# nn_model.save('10.2.DNN_model_MNIST_1st_layer_784units_20epochs.keras')

# // deletes the existing model
# del nn_model

# // returns a compiled model
nn_model = load_model('10-2.DNN_model_MNIST_1st_layer_784units_20epochs.keras')

#@"""



#### Plotting the history (Accuracy and Error plots) ** including the Validation
##"""

## Accuracy plot
'''
acc_plot = plt.plot(nn_model_fit.history['accuracy'], label= "train accuracy" )
val_acc_plot = plt.plot(nn_model_fit.history['val_accuracy'], label="val accuracy" )

plt.xlabel('epochs')
# plt.legend(['accuracy'] )
plt.legend()
plt.title('Accuracies')
plt.show()
'''

## Error plot
'''
loss_plot = plt.plot(nn_model_fit.history['loss'], label = 'train loss' )
val_loss_plot = plt.plot(nn_model_fit.history['val_loss'], label='val loss')

plt.xlabel('epochs')
# plt.legend(['loss'] )
plt.legend()
plt.title('Losses')
plt.show()
'''

## both plots
'''
plt.plot(nn_model_fit.history['accuracy'], color='blue', label='accuracy')
plt.plot(nn_model_fit.history['loss'], color='red', label='loss')
plt.xlabel("number of epochs")
plt.ylabel("percentage")
plt.title("Accuracy and Loss measurements")
plt.legend()
plt.show()
'''

#@"""



#### testing the model
##"""

score = nn_model.evaluate(images_test_N, labels_test_OHE, verbose=0)
# // nn_model.evaluate returns a list
print(type(score) )
# // <class 'list'>

# // verbose = 0 to not show the details
print("Test data loss evaluation: ", score[0])  # // Test loss evaluation: 0.22091379761695862
print("Test data accuracy evaluation: ", score[1])  # // Test accuracy evaluation: 0.9381999969482422
print(score)  # // [0.22091379761695862, 0.9381999969482422]


# // second model
print("Test loss evaluation: ", score[0])  # // Test loss evaluation: 0.15839534997940063
print("Test accuracy evaluation: ", score[1])  # // Test accuracy evaluation:  0.9761999845504761
print(score)  # // [0.15839534997940063, 0.9761999845504761]

# // with CNN, we can easily reach 98 or 99% accuracy
# // another advantage of CNN is how much efficient they are at classifying much larger and colorful images.

#@"""




