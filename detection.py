import torch
import torch.nn as nn

import cv2
import numpy as np

image_path = "geoclidean/constraints/concept_ccc/test/in_5_fin.png"
image_path = "geoclidean/constraints/concept_cccl/train/1_fin.png"
# read the image

img = cv2.imread(image_path,cv2.IMREAD_COLOR)
# convert to gray scale

gray_blurred = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
# blur using the 3x3 kernel

detected_circles = cv2.HoughCircles(gray_blurred,
                                    cv2.HOUGH_GRADIENT,1,40,param1 = 50,
                                    param2 = 30, minRadius = 2, maxRadius = 150)

edges = cv2.Canny(gray_blurred,50,200,None,apertureSize = 3)
maxLineGap = 100
detected_lines= lines = cv2.HoughLinesP(edges,1,np.pi/180,10,1,maxLineGap)

if detected_lines is not None:
    for x1,y1,x2,y2 in detected_lines[0]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow("Detected Circle",img)
cv2.waitKey(0)

output_circles = []

# draw circles that are detected
if detected_circles is not None:

    # convert the circle parameters a,b and r to integers
    detected_circles = np.uint16(np.around(detected_circles))

  
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
  
        # Draw the circumference of the circle.
        cv2.circle(img, (a, b), r, (0, 255, 0), 2)
  
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
        output_circles.append([a,b,r])
    
cv2.imshow("Detected Circle",img)
cv2.waitKey(0)

def parse_raw_lines(image):
    return 0

def parse_raw_circles(image):
    return 0

def parse_raw_points(image):
    return 0

print(output_circles)