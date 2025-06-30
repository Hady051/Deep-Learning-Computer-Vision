
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

#### Initial steps - making and plotting the points
##"""

number_of_points = 50

np.random.seed(3) # // to make the random variables constant

random_x1_values = np.random.normal(10, 2, number_of_points)  # // points according to the horizontal axis
# print(random_x1_values)
random_x2_values = np.random.normal(12, 2, number_of_points)  # // points according to the vertical axis

top_region_points = np.array([random_x1_values, random_x2_values]).T
print(top_region_points)   # // this way we will have each row has 2 elements (points location)->(x1, x2)

## bottom region
random_x1_values_2 = np.random.normal(5, 2, number_of_points)
random_x2_values_2 = np.random.normal(6, 2, number_of_points)
bottom_region_points = np.array([random_x1_values_2, random_x2_values_2]).T

#@"""



#### display the points
##"""
_, axis = plt.subplots(figsize=(4, 4) )
# // this returns a tuple, so we can unpack it to fig, axis, since we won't use fig, leave it blank
axis.scatter(top_region_points[:, 0], top_region_points[:, 1], color='r')
# // the horizontal points(coordinates) and the vertical points(coordinates), the color
axis.scatter(bottom_region_points[:, 0], bottom_region_points[:, 1], color='b')
# // the horizontal points(coordinates) and the vertical points(coordinates), the color
plt.show()

#@"""