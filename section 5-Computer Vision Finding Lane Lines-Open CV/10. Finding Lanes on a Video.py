
## Optimization but cutting unnecessary bits out
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
    polygon = np.array( [ [ (200, image_bottom), (1100, image_bottom), (550, 250) ] ]  )
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
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10) # // now the line is merged
            # // until now this shows the lines in the masked image but in a correct position thanks to Hough function
            #cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 17)
            #cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 17)
    return line_image


# // this was done at the end of average_slope_intercept function
def make_coordinates(image, line_parameters):
    #slope, intercept = line_parameters
    try:                                        # // ***solution for ERROR when there is a gap between lines
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0.01, 0.01
    # // with this part above, I don't need the if statement "if len()"

    y1 = image.shape[0]                         # // 704
    y2 = int(y1 * (3/5) )                       # // 422
    # // can change it to (2/5) to make the lines taller
    x1 = int( (y1 - intercept) / slope)           # // depends on the slope and y-intercept
    x2 = int( (y2 - intercept) / slope)           # // depends on the slope and y-intercept
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    #if lines is None:    # // **this is among the solution of the video not working midway
    #    return None      # // it doesn't matter much

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        #print(parameters)  # // this shows the slope as the first element and the y-intercept as the second.
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append( (slope, intercept) )
        else:
            right_fit.append( (slope, intercept) )
    #print(left_fit) # // this shows all the slopes and y-intercepts of the lines on the left
    #print(right_fit) # // this shows all the slopes and y-intercepts of the lines on the right

    #if len(left_fit) and len(right_fit):  # // ** among the solution of video stopping

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    average_LeftLine = make_coordinates(image, left_fit_average)
    average_RightLine = make_coordinates(image, right_fit_average)

    averaged_lines = [average_LeftLine, average_RightLine]
    return averaged_lines   # // this was changed, but it doesn't do anything different

    #return np.array([average_LeftLine, average_RightLine]) # // the original


#@"""



## Capturing a video and showing the lines on it-the relatively course code
"""

video_path = 'test2.mp4'
video = cv2.VideoCapture(video_path)

while video.isOpened():
    _, pic_frame = video.read()            # // replace lane_image by frame

    # // edge detected image/canny
    edgeDetected_image = canny(pic_frame)

    # // masked_lane_image
    masked_lane_image = region_of_interest(edgeDetected_image)

    # // the lines by Hough Transform
    lines = cv2.HoughLinesP(masked_lane_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    # // the averaged lines optimization
    averaged_lines = average_slope_intercept(pic_frame, lines)

    # // the optimized average lines
    avg_lines_image = display_lines(pic_frame, averaged_lines)

    # // the lane image with lines
    lane_image_with_lines = cv2.addWeighted(pic_frame, 0.8, avg_lines_image, 1, 1)

    # // showing the video
    cv2.imshow('Lane Video with the Lines', lane_image_with_lines)
    # // ** to show each frame in a 1 millisecond,
    # // we will add an if statement to add keyboard key to stop the video manually
    if cv2.waitKey(1) & 0xFF == ord('s'): # // type 's' to close
        break

# // these 2 functions are used to close the video properly
video.release()
cv2.destroyAllWindows()

"""



## ** Solution for the unpack error, with modifications above as well
##"""

# // ** an error happens midway when the left line on the road is changed into parts
# // "TypeError: cannot unpack non-iterable numpy.float64 object"
# // this is the solution for it
# // among the solution is the if statement "if len(left_fit) and len(right_fit):"
# in the average slope function
# // ** the better solution is the part of try and except in make_coordinates function,
# if you use it, you can even remove the if statement


video_path = 'test2.mp4'
video = cv2.VideoCapture(video_path)

while ( video.isOpened() ):
    ute, pic_frame = video.read()            # // replace lane_image by frame
    
    if ute == True:
        # // the edge detected image
        edgeDetected_image = canny(pic_frame)
        
        # // the masked lane image
        masked_lane_image = region_of_interest(edgeDetected_image)

        # // the lines by Hough Transform
        lines = cv2.HoughLinesP(masked_lane_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)

        # // the averaged lines optimization
        averaged_lines = average_slope_intercept(pic_frame, lines)
    
        # // displaying the optimized average lines
        avg_lines_image = display_lines(pic_frame, averaged_lines)

        # // the lane image with lines
        lane_image_with_lines = cv2.addWeighted(pic_frame, 0.8, avg_lines_image, 1, 1)

        # // showing the video
        cv2.imshow('Lane Video with the Lines', lane_image_with_lines)
        # // ** to show each frame in a 1 millisecond,
        # // we will add an if statement to add keyboard key to stop the video manually
        if cv2.waitKey(1) & 0xFF == ord('s'): # // type 's' to close
            break
    else:
        break

# // these 2 functions are used to close the video properly
video.release()
cv2.destroyAllWindows()

#@"""