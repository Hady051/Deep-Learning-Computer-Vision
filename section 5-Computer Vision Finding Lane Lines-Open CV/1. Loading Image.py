

## Loading an image
##"""

import cv2

# path = 'D:\\coding projects\\PyCharm projects\\new projects\\S-D-C-2'
#                   '\\section 5-Computer Vision Finding Lane Lines-Open CV\\Image\\test_image.jpg'
# image = cv2.imread(path)


# path = r'D:\coding projects\PyCharm projects\new projects\S-D-C-2\section 5-Computer Vision Finding Lane Lines-Open CV\Image\test_image.jpg'
# image = cv2.imread(path)


image = cv2.imread('test_image.jpg') # / this is the image inside the same folder,
# / so we can just type the name of the pic without needing
# / to type the absolute path

cv2.imshow('test', image)
cv2.waitKey(0)

#@"""