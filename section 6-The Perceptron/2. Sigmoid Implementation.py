
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
    return 1/(1 + np.exp(-input) )

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
# print(top_region_points)   # // this way we will have each row has 2 elements (points location)->(x1, x2)

## bottom region
random_x1_values_2 = np.random.normal(5, 2, number_of_points)
random_x2_values_2 = np.random.normal(6, 2, number_of_points)
bottom_region_points = np.array([random_x1_values_2, random_x2_values_2, bias_every_point]).T

## all points
all_points = np.vstack((top_region_points, bottom_region_points))
# print(all_points)

## Line
w1 = -0.2
w2 = -0.35
b = 3.5  # // bias value
line_parameters = np.matrix([w1, w2, b]).T # // took the transpose so we can multiply it by all_points
x1 = np.array([bottom_region_points[:, 0].min() - 2, top_region_points[:, 0].max() + 2 ] )
# // here I took the min from the horizontal points of the bottom region,
# // and the max from the horizontal points of the top region

# // w1x1 + w2x2 + b = 0
# // x2 = -b/w2 - (w1 * x1)/w2
x2 = -b/w2 - (w1 * x1)/w2
# print(x1, x2)
# // the array on the left has the x1 points on the x1(x) axis,
# // the right array has the corresponding x2 points on the x2 (y) axis
# print(line_parameters.shape) # // (3,1)
# print(all_points.shape)      # // (100,3)

## perceptron (multiplying the values)
linear_combination = all_points * line_parameters
# print(linear_combination)

scores = sigmoid(linear_combination)
print(scores)

#@"""



#### Display the points
##"""
_, axis = plt.subplots(figsize=(4, 4) )

axis.scatter(top_region_points[:, 0], top_region_points[:, 1], color='r')
axis.scatter(bottom_region_points[:, 0], bottom_region_points[:, 1], color='b')
draw_line(x1, x2)
plt.show()

#@"""

