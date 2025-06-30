#### Importing the libraries
##"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

#@"""



#### Initial steps-making and plotting the points
##"""

number_of_points = 500
np.random.seed(3)

# bias_every_point = np.ones(number_of_points)

## top region points
random_x1_values_top = np.random.normal(13, 2, number_of_points)
random_x2_values_top = np.random.normal(12, 2, number_of_points)
top_region_points = np.array([random_x1_values_top, random_x2_values_top]).T
# print(top_region_points.shape)   # // (2, 50), after transpose-> (50, 2)

## bottom region points
random_x1_values_bot = np.random.normal(8, 2, number_of_points)
random_x2_values_bot = np.random.normal(6, 2, number_of_points)
bottom_region_points = np.array([random_x1_values_bot, random_x2_values_bot]).T

## all points
all_points = np.vstack( (top_region_points, bottom_region_points) )

## Label
# y = np.array([np.zeros(number_of_points), np.ones(number_of_points) ] ).reshape(number_of_points * 2, 1)
# print(y.shape)       # // (2, 50), after reshaping -> (100, 1)
# y = np.array([np.zeros(number_of_points), np.ones(number_of_points)]).T
# print(y.shape)         # // (50, 2)
# // for y (label) to be the same size as all points (both = 100), we use np.append()
y = np.matrix(np.append(np.zeros(number_of_points), np.ones(number_of_points) ) ).T
# print(y.shape)  # // (100, 1)

#@"""



#### Displaying the points
"""

plt.scatter(top_region_points[ : , 0], top_region_points[ : , 1], color='r')
plt.scatter(bottom_region_points[ : , 0], bottom_region_points[ : , 1], color = 'b')

"""



#### sequential model
##"""

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

nn_model = Sequential()

nn_model.add(Dense(units=1, input_shape=(2,), activation='sigmoid') )

# adam = Adam(learning_rate=0.01)
adam = Adam(learning_rate=0.03)
nn_model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'] )

# nn_model_fit = nn_model.fit(x=all_points, y=y, verbose=1, batch_size=50, epochs=100, shuffle='true')
# // or h (the common variable for model.fit)

nn_model_fit = nn_model.fit(x=all_points, y=y, verbose=1, batch_size=15, epochs=20, shuffle='true')

#@"""



#### Plotting the accuracy chart/curve
"""

plt.plot(nn_model_fit.history['accuracy'])
plt.xlabel('epochs')
plt.legend(['accuracy'])
plt.title('Accuracy')

"""



#### Plotting the loss (error) curve
"""

plt.plot(nn_model_fit.history['loss'])
plt.xlabel('epochs')
plt.legend(['loss'])
plt.title('Loss')

"""



#### plotting both accuracy and loss together (4 subplots)
"""

figure, axis = plt.subplots(2, 2)


axis[0, 0].plot(nn_model_fit.history['accuracy'])
axis[0, 0].set_title("accuracy")
plt.xlabel('epoch')
plt.legend(['accuracy'])


axis[0, 1].plot(nn_model_fit.history['loss'])
axis[0, 1].set_title("loss")
plt.xlabel('epoch')
plt.legend(['loss'])

plt.show()

"""



#### plotting both accuracy and loss together (2 subplots)
"""

figure, axis = plt.subplots(1, 2)

axis[0].plot(nn_model_fit.history['accuracy'])
axis[0].set_title("accuracy")
plt.xlabel('epoch')
plt.legend(['accuracy'])

axis[1].plot(nn_model_fit.history['loss'])
axis[1].set_title("loss")
plt.xlabel('epoch')
plt.legend(['loss'])

plt.show()

"""



#### plotting both accuracy and loss(error) simultaneously in one plot
##"""

## Plotting the curves
plt.plot(nn_model_fit.history['accuracy'], color='b', label='accuracy')
plt.plot(nn_model_fit.history['loss'], color='r', label='loss/error')

## naming the x-axis, y-axis and the whole graph
plt.xlabel('epochs')
plt.ylabel('values')
plt.title('Accuracy and Loss measurements')

## Adding a legend to help us recognize the curve according to it's color
plt.legend()

## to load the display window
plt.show()


#@"""