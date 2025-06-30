import cv2
import numpy as np

## loading the image
##"""

path = 'test_image.jpg'

image = cv2.imread(path)

#cv2.imshow('Original Image', image)
#cv2.waitKey(0)

#@"""



## grayscale
##"""

image_2 = np.copy(image)
gray_image = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)

#cv2.imshow('Grayscale Image', gray_image)
#cv2.waitKey(0)

#@"""



## smoothening
##"""

smooth_image = cv2.GaussianBlur(gray_image, (5,5), 0)

#cv2.imshow('Smooth Image', smooth_image)
#cv2.waitKey(0)

#@"""



## edge detection/ Canny function
##"""

# // canny method automatically applies gaussian blur, so we didn't need the smoothened code

canny_image = cv2.Canny(smooth_image, 50, 150)

cv2.imshow('Edge detected image', canny_image)
cv2.waitKey(0)

#@"""