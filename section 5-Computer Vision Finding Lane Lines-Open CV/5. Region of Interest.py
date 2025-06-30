import cv2
import numpy as np

## loading image
##"""

#path = 'test_image.jpg'

#image = cv2.imread(path)

#cv2.imshow('Original Image', image)
#cv2.waitKey(0)

#@"""



## grayscale
"""

image_2 = np.copy(image)
gray_image = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)

#cv2.imshow('Grayscale Image', gray_image)
#cv2.waitKey(0)

"""



## smoothening
"""

smooth_image = cv2.GaussianBlur(gray_image, (5,5), 0)

#cv2.imshow('Smooth Image', smooth_image)
#cv2.waitKey(0)

"""



## edge detection/ Canny function
"""

canny_image = cv2.Canny(smooth_image, 50, 150)

cv2.imshow('Edge detected image', canny_image)
cv2.waitKey(0)

"""



## Region of interest 1
"""

# / just added the previous 3 steps in a function for easier use


def canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    smooth_image = cv2.GaussianBlur(gray_image, (5,5), 0)
    canny_image = cv2.Canny(smooth_image, 50, 150)
    return canny_image


path = 'test_image.jpg'
image = cv2.imread(path)
print(image.shape)  # / (704, 1279, 3)

image_2 = np.copy(image)
edgeDetected_image = canny(image_2)

cv2.imshow('Edge Detected Image', edgeDetected_image)
cv2.waitKey(0)

"""



## Matplotlib example
"""
import matplotlib
matplotlib.use("QtAgg")
#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

# / creating dataset
cars = ['Audi', 'Bmw', 'Ford', 'Dodge', 'Porsche', 'Mercedes']
data = [23, 17, 35, 29, 12, 41]

# / Creating plot
figure = plt.figure(figsize=(10, 7) )
plt.pie(data, labels=cars)

# / show plot
plt.show()

"""



## Matplotlib
"""

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

path = 'test_image.jpg'
image = cv2.imread(path)

image_2 = np.copy(image)
edgeDetected_image = canny(image_2)


plt.imshow(edgeDetected_image)  # / don't run python console to work

plt.show()

"""




## Region of interest 2
##"""

# // just added the previous 3 steps in a function for easier use


def canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    smooth_image = cv2.GaussianBlur(gray_image, (5,5), 0)
    canny_image = cv2.Canny(smooth_image, 50, 150)
    return canny_image

# // making a function to identify the region of interest


def region_of_interest(image):
    image_bottom = image.shape[0] # // 704 biggest height
    #triangle = np.array([ (200, image_bottom), (1100, image_bottom), (550, 250) ])
    # // added square brackets so that it can be an array because of the requirement of the fillpoly() function
    polygon = np.array( [ [ (200, image_bottom), (1100, image_bottom), (550, 250) ] ] )
    mask_image = np.zeros_like(image) # // black image(mask) to put the polygon in
    cv2.fillPoly(mask_image, polygon, 255)
    return mask_image


path = 'test_image.jpg'
image = cv2.imread(path)
print(image.shape)  # // (704, 1279, 3)

image_2 = np.copy(image)
edgeDetected_image = canny(image_2)

# // showing the edge detected image
# cv2.imshow('Edge DetectedImage', edgeDetected_image)
#cv2.waitKey(0)

# // showing the ROI image
cv2.imshow('Region Of Interest pic', region_of_interest(edgeDetected_image) )
cv2.waitKey(0)

#@"""
