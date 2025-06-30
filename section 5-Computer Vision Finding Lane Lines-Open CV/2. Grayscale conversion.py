

## Grayscale
##"""

import cv2
import numpy as np

path = 'test_image.jpg'
image = cv2.imread(path)
#cv2.imshow('Original Pic', image)
#cv2.waitKey(0)

lane_image = np.copy(image)
gray_image = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
cv2.imshow('Grayscaled Image', gray_image)
cv2.waitKey(0)

#@"""