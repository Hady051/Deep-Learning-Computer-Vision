

## filtering the image
##"""

import cv2
import numpy as np

# / original image
path = 'test_image.jpg'
image = cv2.imread(path)

#cv2.imshow('Original Image', image)
#cv2.waitKey(0)

# / grayscale image
image_2 = np.copy(image)
gray_image = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)

#cv2.imshow('Grayscale Image', gray_image)
#cv2.waitKey(0)

# / Filtered image
smooth_image = cv2.GaussianBlur(gray_image, (5,5), 0)

cv2.imshow('Smoothened Image', smooth_image)
cv2.waitKey(0)

#@"""