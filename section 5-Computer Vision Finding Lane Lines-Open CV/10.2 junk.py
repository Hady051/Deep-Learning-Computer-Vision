
## // just made that to try my older solutions from SDC 1


import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(image, line_parameters):
    try:                                                  ## solution for ERROR when there is a gap between lines
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0.01, 0.01

    # slope, intercept = line_parameters

    # print(image.shape)                it shows the image(array) width(704), height(1279), no.of channels(3)
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))

    x1 = int( (y1 - intercept) / slope )
    x2 = int( (y2 - intercept) / slope )

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    # for average of left line for lane
    left_fit = []
    # for average of left line for lane
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit( (x1, x2), (y1, y2), 1 )
        # print(parameters)           it shows the slope as first element and y-intercept as second element
        slope = parameters[0]
        intercept = parameters[1]
        # lines in the left has -ve slope and lines in right has +ve slope because of changes in y and x
        if slope < 0:
            left_fit.append( (slope, intercept) )
        else:
            right_fit.append( (slope, intercept) )

    #print(left_fit)                      # list of slopes and y-intercepts of the left side
    #print(right_fit)                     # same for right side
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])


def canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(blur_image, 50, 150)
    return canny_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            #print(line)
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line( line_image, (x1, y1), (x2, y2), (255, 0, 0), 15 )                 # this shows a line segment
            # connecting 2 points with BGR 255, 0, 0 is blue, then line thickness
    return line_image


def region_of_interest(image):
    # height = image.shape[0]   # instead of 700              # image bottom y = 0
    polygon = np.array([
        [ (200, 704), (1100, 704), (550, 250) ]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)                # so that I get the white(lane) only
    return masked_image


image_path = r'D:\coding projects\PyCharm projects\new projects\S-D-C\section 5\Image\test_image.jpg'
image = cv2.imread(image_path)

lane_image = np.copy(image)
#canny_image = canny(lane_image)

#cropped_image = region_of_interest(canny_image)

#detected_lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # 120


#averaged_lines = average_slope_intercept(lane_image, detected_lines)


#line_image = display_lines(lane_image, averaged_lines)


#combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)


#cv2.imshow( " image result 1", line_image)

#cv2.imshow( "image result 2", combo_image )
#cv2.waitKey(0)
#cv2.destroyAllWindows()

## video

vid_path = 'D:\\coding projects\\PyCharm projects\\new projects\\S-D-C\\section 5\\video test\\test2.mp4'
vid = cv2.VideoCapture(vid_path)

while ( vid.isOpened() ):                 # replace lane_image by frame
    ute, pic_frame = vid.read()

    if ute == True:

        canny_image = canny(pic_frame)

        cropped_image = region_of_interest(canny_image)

        detected_lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40,
                                         maxLineGap=5)  # 120

        averaged_lines = average_slope_intercept(pic_frame, detected_lines)

        line_image = display_lines(pic_frame, averaged_lines)

        combo_image = cv2.addWeighted(pic_frame, 0.8, line_image, 1, 1)


        cv2.imshow("video result 2", combo_image)

        if cv2.waitKey(1) & 0xff == ord('s'):
            break

    else:
        break

vid.release()

cv2.destroyAllWindows()
