

## Hough Transform
##"""


import cv2
import numpy as np


def canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    smooth_image = cv2.GaussianBlur(gray_image, (5,5), 0)
    canny_image = cv2.Canny(smooth_image, 50, 150)
    return canny_image


def region_of_interest(image):
    image_bottom = image.shape[0] # // 704 biggest height
    #triangle = np.array([ (200, image_bottom), (1100, image_bottom), (550, 250) ])
    # // added square brackets so that it can be an array because of the requirement of the fillpoly() function
    polygon = np.array( [ [ (200, image_bottom), (1100, image_bottom), (550, 250) ] ] )
    mask_image = np.zeros_like(image) # // black image(mask) to put the polygon in
    cv2.fillPoly(mask_image, polygon, 255) # // now the polygon is in the mask, in white color
    masked_image = cv2.bitwise_and(image, mask_image)
    return masked_image


def display_lines(image, lines):
    # // create a mask picture for the lines
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            #print(line)  # // this shows each line that isn't none, is 2D, we need to change it to 1D
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10) # // now the line is merged
            # // until now this shows the lines in the masked image but in a correct position thanks to Hough function


    return line_image



path = 'test_image.jpg'
image = cv2.imread(path)
#print(image.shape)  # // (704, 1279, 3)

lane_image = np.copy(image)
edgeDetected_image = canny(lane_image)

# // showing the edge detected image
#cv2.imshow('Edge DetectedImage', edgeDetected_image)
#cv2.waitKey(0)

# // just putting the result of ROI in a function
masked_lane_image = region_of_interest(edgeDetected_image)

# // showing the new ROI image
#cv2.imshow('Masked Lane pic', masked_lane_image)
#cv2.waitKey(0)

# // the lines by Hough Transform
lines = cv2.HoughLinesP(masked_lane_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

# // the masked image with the lines, taking the same parameters of the og pic
lines_image = display_lines(lane_image, lines)

# // showing the masked line image
#cv2.imshow('Lines masked pic', lines_image)
#cv2.waitKey(0)

# // we then blend the blue lines to the actual image, and since the rest of the mask image is black, we can use
# // addWeighted function since the intensity of the pixels in the mask image except for the lines is 0 and adding with
# // 0 doesn't make any difference.

lane_image_with_lines = cv2.addWeighted(lane_image, 0.8, lines_image, 1, 1)


# // showing the lane with the lines image
cv2.imshow('Lane with the Lines pic', lane_image_with_lines)
cv2.waitKey(0)


#@"""