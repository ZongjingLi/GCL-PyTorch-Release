import torch
import torch.nn as nn

import cv2
import numpy as np

image_path = "geoclidean/constraints/concept_ccc/test/in_5_fin.png"
image_path = "geoclidean/constraints/concept_cccl/train/1_fin.png"
image_path = "geoclidean/constraints/concept_lll/train/3_fin.png"
image_path = "geoclidean/elements/concept_ang_bisector/train/5_fin.png"
image_path = "geoclidean/elements/concept_diameter/train/{}_fin.png"
image_path = "geoclidean/elements/concept_{}/{}/{}_fin.png"
#image_path = "geoclidean/constraints/concept_llcc/train/4_fin.png"
# read the image

def raw_parse(image_path):
    img = cv2.imread(image_path,cv2.IMREAD_COLOR)
    # convert to gray scale

    gray_blurred = cv2.cvtColor(img ,cv2.COLOR_RGBA2GRAY)
    # blur using the 3x3 kernel

    detected_circles = cv2.HoughCircles(gray_blurred,
                                    cv2.HOUGH_GRADIENT,1,40,param1 = 60,
                                    param2 = 30, minRadius = 4, maxRadius = 150)

    edges = cv2.Canny(gray_blurred,25, 250)
    detected_lines= cv2.HoughLinesP(edges, 1, np.pi/180, 60, maxLineGap=200)
    #detected_lines= cv2.HoughLines(gray_blurred, 1, np.pi/180,700)

    output_lines = []

    if detected_lines is not None:
        for line in detected_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 256), 2)
            output_lines.append([(x1,y1),(x2,y2)])

    #cv2.imshow("Detected Lines",img)
    #cv2.waitKey(0)

    output_circles = []

    # draw circles that are detected
    if detected_circles is not None:

        # convert the circle parameters a,b and r to integers
        detected_circles = np.uint16(np.around(detected_circles))

        # namomo, adjust lines
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
  
            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 3)
  
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
            output_circles.append([a,b,r])
    
    cv2.imshow("Detected Circle",img)
    cv2.waitKey(0)
    return output_lines,output_circles

def parse_raw_lines(image):
    return 0

def parse_raw_circles(image):
    return 0

def parse_raw_points(image):
    return 0

names = ["ang_bisector","angle","diameter","eq_t","oblong","parallel_l",
        "perp_bisector","quadrilateral","radii","rectilinear","rhomboid",
        "rhombus","right_ang_t","segment","sixty_ang","square","triangle"]

def reveal_all_concepts(names,split = "train0"):
    image_path = "geoclidean/elements/concept_{}/{}/{}_fin.png"
    for name in names:
        for i in range(5):
            a,b = raw_parse(image_path.format(name,split,i+1))
            print(a,b)

def reveal_all_constraints(names,split = "train"):
    image_path = "geoclidean/constraints/concept_{}/{}/{}_fin.png"
    for name in names:
        for i in range(5):
            a,b = raw_parse(image_path.format(name,split,i+1))
            print(a,b)

cons = ["ccc","cccc","cccl","ccl","cct","clcc","clcl","cll","clll","clt","lcc",
"llcc","lll","lllc","llll","llt","tcc","tcl","tll"]

reveal_all_constraints(cons)

reveal_all_concepts(names)
