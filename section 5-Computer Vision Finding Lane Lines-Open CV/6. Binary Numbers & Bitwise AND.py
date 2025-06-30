

## Bitwise AND
##"""


import cv2
import numpy as np


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
    cv2.fillPoly(mask_image, polygon, 255) # // now the polygon is in the mask, with white color
    masked_image = cv2.bitwise_and(image, mask_image)
    return masked_image


path = 'test_image.jpg'
image = cv2.imread(path)
print(image.shape)  # // (704, 1279, 3)

image_2 = np.copy(image)
edgeDetected_image = canny(image_2)

# // just putting the result of ROI in a function
masked_lane_image = region_of_interest(edgeDetected_image)

# // showing the edge detected image
#cv2.imshow('Edge DetectedImage', edgeDetected_image)
#cv2.waitKey(0)

# // showing the new ROI image
cv2.imshow('Masked Lane pic', masked_lane_image)
cv2.waitKey(0)

#@"""


