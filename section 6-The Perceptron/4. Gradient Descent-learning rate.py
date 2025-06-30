import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

#### function-drawing the line
##"""

def draw_line(x, y):
    line = plt.plot(x, y)

#@"""



#### function-sigmoid
##"""

def sigmoid(input):
    return 1 / (1 + np.exp(-input) )

#@"""



#### Function-calculating the error
##"""

def calculate_error(line_parameters, points, label):
    m = points.shape[0] # // returns a tuple of our arrays dimension
    linear_combination = points * line_parameters
    p = sigmoid(linear_combination)
    # // the CE-error equation, y = label
    cross_entropy = -1 * (np.log(p).T * label + np.log(1 - p).T * (1 - label))
    # print("cross_entropy", cross_entropy.shape)
    ce_error = cross_entropy / m
    print("the cross entropy error is: ", ce_error * 100, "%")
    return ce_error

#@"""



#### Function- gradient descent
##"""

def gradient_descent(line_parameters, points, label, learning_rate, loops_number):
    m = points.shape[0]
    for i in range(loops_number):
        p = sigmoid(points * line_parameters)
        gradient_descent = (points.T * (p - label) ) * (1 / m) * learning_rate
        line_parameters = line_parameters - gradient_descent
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)

        x1 = np.array([points[:, 0].min() - 2, points[:, 0].max() + 2] )
        x2 = -b/w2 - (w1 * x1)/w2

        print(gradient_descent)

    draw_line(x1, x2)
    print("Latest error is: ", gradient_descent)
    return gradient_descent

#@"""



#### Initial steps - making and plotting the points
##"""

number_of_points = 50

np.random.seed(3)

## bias
bias_every_point = np.ones(number_of_points)

random_x1_values = np.random.normal(10, 2, number_of_points)  # // points according to the horizontal axis
random_x2_values = np.random.normal(12, 2, number_of_points)  # // points according to the vertical axis
top_region_points = np.array([random_x1_values, random_x2_values, bias_every_point]).T

## bottom region
random_x1_values_2 = np.random.normal(5, 2, number_of_points)
random_x2_values_2 = np.random.normal(6, 2, number_of_points)
bottom_region_points = np.array([random_x1_values_2, random_x2_values_2, bias_every_point]).T

## all points
all_points = np.vstack((top_region_points, bottom_region_points))

## Line
line_parameters = np.matrix([np.zeros(3)]).T # // took the transpose ,so we can multiply it by all_points

# x1 = np.array([bottom_region_points[:, 0].min() - 2, top_region_points[:, 0].max() + 2] ) # // [:, 0] means horizontal
# x2 = -b/w2 - (w1 * x1)/w2
# // copied and pasted in the gradient function

## Label
y = np.array([np.zeros(number_of_points), np.ones(number_of_points)]).reshape(number_of_points * 2, 1)
# // we basically make the first 50 points zeros and the second 50 points ones
# print(y)

#@"""



#### Display the points
##"""
_, axis = plt.subplots(figsize=(4, 4) )

axis.scatter(top_region_points[:, 0], top_region_points[:, 1], color='r')
axis.scatter(bottom_region_points[:, 0], bottom_region_points[:, 1], color='b')
# gradient_descent(line_parameters, all_points, y, learning_rate=0.06, loops_number=600)
# new_line_parameters, new_all_points, new_y = gradient_descent(line_parameters, all_points, y, 0.02, 2000)
gradient_descent(line_parameters, all_points, y, 0.06, 6000)

plt.show()

#@"""


#### Error
##"""

# calculate_error(new_line_parameters, new_all_points, new_y)

#@"""