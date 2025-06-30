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

nn_model_fit = nn_model.fit(x=all_points, y=y, verbose=1, batch_size=10, epochs=20, shuffle='true')

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
"""

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

"""



#### function-plotting the decision boundary
##"""

def plot_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[ : , 0]) - 1, max(X[ : , 0]) + 1, 50)
    y_span = np.linspace(min(X[ : , 1]) - 1, max(X[ : , 1]) + 1, 50)
    # print("x_span = ", x_span)     # // horizontal
    # print(f"y_span =  {y_span}")      # // vertical
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()  # // to flatten them from 2D to 1D
    grid = np.c_[xx_, yy_]
    prediction_function = model.predict(grid)
    prediction_function_reshaped = prediction_function.reshape(xx.shape)
    plt.contourf(xx, yy, prediction_function_reshaped)

#@"""



#### plotting the decision boundary prediction
##"""

## plotting the contour
plot_decision_boundary(all_points, y, nn_model)

## plotting the points
plt.scatter(top_region_points[ : , 0], top_region_points[ : , 1], color='r')
plt.scatter(bottom_region_points[ : , 0], bottom_region_points[ : , 1], color = 'b')

## plotting the point to be predicted
point_x = 7.5
point_y = 5
point = np.array([[point_x, point_y] ] )
prediction = nn_model.predict(point) * 100

prediction_word = ''
if prediction >= 70:
    prediction_word = 'Healthy'
elif prediction < 70 and prediction > 30:
    prediction_word = 'moderatly healthy'
else:
    prediction_word = 'Unhealthy'

prediction = str(prediction) + '%'

plt.plot([point_x], [point_y], marker='o', markersize=10, color='orange')
print(f"The prediction for the point is: {prediction}")
print(f"The patient's status prediction is that he is: {prediction_word}")


#@"""