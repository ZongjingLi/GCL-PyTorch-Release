import torch
import torch.nn as nn

import cv2
import numpy as np

image_path = "geoclidean/constraints/concept_ccc/test/in_5_fin.png"
image_path = "geoclidean/constraints/concept_cccl/train/1_fin.png"
image_path = "geoclidean/constraints/concept_lll/train/2_fin.png"
# read the image

img = cv2.imread(image_path,cv2.IMREAD_COLOR)
# convert to gray scale

gray_blurred = cv2.cvtColor(img ,cv2.COLOR_RGBA2GRAY)
# blur using the 3x3 kernel

detected_circles = cv2.HoughCircles(gray_blurred,
                                    cv2.HOUGH_GRADIENT,1,40,param1 = 50,
                                    param2 = 30, minRadius = 2, maxRadius = 150)

edges = cv2.Canny(gray_blurred,75, 150)
#detected_lines= cv2.HoughLinesP(edges, 1, np.pi/180, 20, maxLineGap=50)
detected_lines= cv2.HoughLines(edges, 1, np.pi/180,90)

output_lines = []

if detected_lines is not None:
    for line in detected_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # x1 stores the rounded off value of (r* cosΘ - 1000 * sinΘ)
        x1 = int(x0 + 1000 * (-b))
        # y1 stores the rounded off value of (r * sinΘ + 1000 * cosΘ)
        y1 = int(y0 + 1000 * (a))
        # x2 stores the rounded off value of (r * cosΘ + 1000 * sinΘ)
        x2 = int(x0 - 1000 * (-b))
        # y2 stores the rounded off value of (r * sinΘ - 1000 * cosΘ)
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("Detected Circle",img)
cv2.waitKey(0)

output_circles = []

# draw circles that are detected
if detected_circles is not None:

    # convert the circle parameters a,b and r to integers
    detected_circles = np.uint16(np.around(detected_circles))

    # namomo, adjust lines
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

print("circles:")
print(output_circles)
print("lines:")
print(output_lines)