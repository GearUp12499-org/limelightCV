import numpy as np
import cv2
import sys
import math
import os


# CONSTATS GO HERE
YELLOW = 0
RED = 1
BLUE = 2

# CENTER_X = 320
# CENTER_Y = 320

# CONVERSIONS GO HERE
inches2px = lambda inches: inches * 96

def pickupable(x, y, w, h, angle):
        if w > h:
            w, h = h, w
            angle += 90
        ta = w * h
        # Check if the point is inside the rotated rect
        # Offset the y-axis by 0.75 inches to account for difference in the distance from the camera to the pickup
        camMidpoint = (320, 320 + inches2px(0.75))
        # Step 1: translate the rect so the center is (0, 0)
        point = (camMidpoint[0] - x, camMidpoint[1] - y)

        # Step 2: rotate everything by -angle around (0, 0)
        theta = np.deg2rad(-angle)
        point = (
            point[0] * np.cos(theta) - point[1] * np.sin(theta),
            point[0] * np.sin(theta) + point[1] * np.cos(theta)
        )
        xprime = point[0]
        yprime = point[1]
        height = int(.48 * 480 * 0.9)
        width = int(.23 * 480 * 0.9)

        # Step 3: check if the point is within the bounds of the non-rotated rect
        inside = False
        if(xprime > width / 2 or xprime < -width / 2 or yprime > height / 2 or yprime < -height / 2):
            inside = False
        else:
            inside = True
        
        return inside

def calcRectValues(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    x, y = rect[0]
    w, h = rect[1]
    angle = rect[2]

    if w > h:
        w, h = h, w
        angle += 90
    return x, y, w, h, angle, box


def detect(img, color):

    # samples_found will contain array of 4-tuples of detected samples
    #   (distance to center of img, x coord of sample, y coord, angle)
    # At the end we will return the one closest to the center
    #    
    samples_found = []
    #print("color = ", color)

    # convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # compute a mask for the color we are looking for
    if color == YELLOW:
        yellow_lower = np.array([20, 100, 100], dtype="uint8")
        yellow_upper = np.array([40, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    if color == RED:
        red_lower = np.array([0, 100, 100], dtype="uint8")
        red_upper = np.array([10, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsv, red_lower, red_upper)

        red_lower = np.array([170, 100, 100], dtype="uint8")
        red_upper = np.array([180, 255, 255], dtype="uint8")
        red_mask2 = cv2.inRange(hsv, red_lower, red_upper)
        mask += red_mask2

    if color == BLUE:
        blue_lower = np.array([100, 100, 100], dtype="uint8") #Changed from 110 to 100 for blue_lower
        blue_upper = np.array([130, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # blur the mask to get rid of stray pixels
    mask = cv2.GaussianBlur(mask, (31,31), 0)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY) #Make any pixel over 200 to 255 and below to 0


    # compute the expected size of a sample relative the
    # size of the image. 
    height = img.shape[0]
    width = img.shape[1]

    target_h = int(.48 * height)   #  / 480
    target_w = int(.23 * height)   # 

    center_x = 320
    center_y = 320

    
    # find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        # for each contour, find the min area rectangle around it
        x, y, w, h, angle, box = calcRectValues(c) # Box - 4 points for rect

        # Ratios of the width and height of the detected rectangle
        # to the expected width and height
        ratio_w = w / target_w
        ratio_h = h / target_h

        # if the rectangle is too small, skip
        # We are checking if the ratio is a perfect 1:1 or 0.75
        if ratio_w < .75 or ratio_h < .75:
            continue # Skips to the next iteration

        # if it is "just right", record it in samples_found[]
        if ratio_w > .75 and ratio_w < 1.25 and ratio_h > .75 and ratio_h < 1.25:
            rect_vals = [x, y, w, h, angle, box]
            # distance from center of image to center of sample
            dist = math.sqrt((x - center_x)**2 + (y - center_y) **2)
            samples_found.append((dist, x, y, angle, rect_vals))
            continue

        # for big contours (> 1.25), pull of the submask covered by this
        # contour and look further
        submask = np.zeros_like(mask)
        cv2.drawContours(submask, [box], -1, 1, cv2.FILLED)
        submask = submask * mask

        kernel = np.ones((5, 5), np.uint8)
        for _ in range(15):
            # keep eroding until it splits apart
            submask = cv2.erode(submask, kernel)
            sub_contours, _ = cv2.findContours(submask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(sub_contours) <= 1:
                continue

            # test of contours in the subimage
            foundOne = False           
            for subc in sub_contours:
                x, y, w, h, angle, box = calcRectValues(subc)
                
                rw = w / target_w
                rh = h / target_h

                if rw > 0.5 and rw < 1.2 and rh > 0.5 and rh < 1.2:
                    rect_vals = [x, y, w, h, angle, box]
                    dist = math.sqrt((x - center_x)**2 + (y - center_y) **2)
                    samples_found.append((dist, x, y, angle, rect_vals))
                    foundOne = True

            # once we have found a good sample, stop eroding
            if foundOne:
                break

    if len(samples_found) == 0:
        return None, None, None, None
    else:
        # Sort to find the sample closest to the center of the image
        # Return the relative offset of the center of sample
        # converting pixels to inches using target_h = 3.5 inches
        
        samples_found = sorted(samples_found)
        x_offset_px = samples_found[0][1] - center_x
        y_offset_px = samples_found[0][2] - center_y

        in_per_px = 3.5 / target_h
        x_offset_in = x_offset_px * in_per_px
        y_offset_in = y_offset_px * in_per_px
        
        return x_offset_in, y_offset_in, samples_found[0][3], samples_found[0][4]


def runPipeline(img, llrobot):
    # First look for a yellow, then look for the team color
    # rectSutff = [x, y, w, h, angle, box]
    # llrobot = [1.0, 0.0, 0.0]
    try:
        if llrobot[0] > .5:
            xOff, yOff, angle, rectStuff = detect(img, YELLOW) # Use rectStuff to see if the sample is pickupable
            if xOff is not None:
                # isPickupable(x, y, w, h, angle)
                isPickupable = pickupable(rectStuff[0], rectStuff[1], rectStuff[2], rectStuff[3], rectStuff[4])
                returnType = 2.0 if isPickupable else 1.0
                # print([returnType, xOff, yOff, angle])
                return np.array([[]]), img, [returnType, xOff, yOff, angle, 0.0, 0.0, 0.0, 0.0]
            
        if llrobot[1] > .5:
            xOff, yOff, angle, rectStuff = detect(img, RED)
            if xOff is not None:
                isPickupable = pickupable(rectStuff[0], rectStuff[1], rectStuff[2], rectStuff[3], rectStuff[4])
                returnType = 2.0 if isPickupable else 1.0
                return np.array([[]]), img, [returnType, xOff, yOff, angle, 0.0, 0.0, 0.0, 0.0]
                     
        if llrobot[2] > .5:
            xOff, yOff, angle, rectStuff = detect(img, BLUE)
            if xOff is not None:
                isPickupable = pickupable(rectStuff[0], rectStuff[1], rectStuff[2], rectStuff[3], rectStuff[4])
                returnType = 2.0 if isPickupable else 1.0
                return np.array([[]]), img, [returnType, xOff, yOff, angle, 0.0, 0.0, 0.0, 0.0]

        return np.array([[]]), img, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]        

    except Exception as e:
        return np.array([[]]), img, [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# The detect program only picks out the first number for color given in the llrobot array and the rest are ignored
# Light Blue Contour - submask rect
# Purple Contour - every contour
# Dark Blue Contour - good contour after erosion
# Green Contour - First try good contour

# Yellow Test Cases to Fix: 109, 104, 113, 122(js move the bobot), 123, 126, 131, 134
# Red Test Cases to Fix: 122
# Blue Test Cases to Fix: none, really