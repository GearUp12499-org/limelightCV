import numpy as np
import cv2
import sys
import math

YELLOW = 0
RED = 1
BLUE = 2

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
    print("color = ", color)

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
        blue_lower = np.array([110, 100, 100], dtype="uint8")
        blue_upper = np.array([130, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # blur the mask to get rid of stray pixels
    mask = cv2.GaussianBlur(mask, (31,31), 0)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)


    # compute the expected size of a sample relative the
    # size of the image. 
    height = img.shape[0]
    width = img.shape[1]

    target_h = int(.54 * height)   # 260 / 480
    target_w = int(.29 * height)   # 140 / 480

    center_x = width / 2
    center_y = height / 2

    
    # find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        # for each contour, find the min area rectangle around it
        x, y, w, h, angle, box = calcRectValues(c)

        ratio_w = w / target_w
        ratio_h = h / target_h

        print("%6.0f %6.0f %6.0f %6.0f %6.2f %6.2f" % (x, y, w, h, ratio_w, ratio_h))

        # if the rectable is too small, skip
        if ratio_w < .75 or ratio_h < .75:
            cv2.drawContours(img, [box], 0, (255, 255, 255), 2)
            continue

        # if it is "just right", record it in samples_found[]
        if ratio_w > .75 and ratio_w < 1.25 and ratio_h > .75 and ratio_h < 1.25:
            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
            dist = math.sqrt((x - center_x)**2 + (y - center_y) **2)
            samples_found.append((dist, x, y, angle))
            continue

        cv2.drawContours(img, [box], 0, (255, 255, 0), 2)

        # for big contours, pull of the submask covered by this
        # contour and look further
        submask = np.zeros_like(mask)
        cv2.drawContours(submask, [box], -1, 1, cv2.FILLED)
        submask = submask * mask

        kernel = np.ones((5, 5), np.uint8)
        for _ in range(10):
            # keep eroding until it splits apart
            submask = cv2.erode(submask, kernel)
            sub_contours, _ = cv2.findContours(submask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(sub_contours) <= 1:
                continue

            # test of contours in the subimage
            foundOne = False           
            for subc in sub_contours:
                submask = cv2.erode(submask, kernel)
                x, y, w, h, angle, box = calcRectValues(subc)
                
                rw = w / target_w
                rh = h / target_h

                if rw > 0.6 and rw < 1.25 and rh > 0.6 and rh < 1.25:
                    cv2.drawContours(img, [box], 0, (0,255,0), 2)
                    dist = math.sqrt((x - center_x)**2 + (y - center_y) **2)
                    samples_found.append((dist, x, y, angle))
                    foundOne = True

            # once we have found a good sample, stop eroding
            if foundOne:
                break
        
    
    cv2.imshow('orig', img)
    cv2.imshow('mask', mask)

    if len(samples_found) == 0:
        return None, None, None
    else:
        # Sort to find the sample closest to the center of the image
        # 
        # Return the relative offset of the center of sample
        # converting pixels to inches using target_h = 3.5 inches
        
        samples_found = sorted(samples_found)
        x_offset_px = samples_found[0][1] - center_x
        y_offset_px = samples_found[0][2] - center_y

        in_per_px = 3.5 / target_h
        x_offset_in = x_offset_px * in_per_px
        y_offset_in = y_offset_px * in_per_px
        
        return x_offset_in, y_offset_in, samples_found[0][3]


def runPipeline(img, llrobot):
    # First look for a yellow, then look for the team color
    try:
        if llrobot[0] > .5:
            x, y, angle = detect(img, YELLOW)
            if x is not None:
                return np.array([[]]), img, [1.0, x, y, angle, 0.0, 0.0, 0.0, 0.0]
            
        if llrobot[1] > .5:
            x, y, angle = detect(img, RED)
            if x is not None:
                return np.array([[]]), img, [1.0, x, y, angle, 0.0, 0.0, 0.0, 0.0]
            
        if llrobot[2] > .5:
            x, y, angle = detect(img, BLUE)
            if x is not None:
                return np.array([[]]), img, [1.0, x, y, angle, 0.0, 0.0, 0.0, 0.0]

        return np.array([[]]), img, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]        

    except Exception as e:
        print(e)
        return np.array([[]]), img, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


    

img = cv2.imread(sys.argv[1])
_, _, llpython = runPipeline(img, [1.0, 1.0, 0.0])

print(llpython[:4])

cv2.waitKey(0)
cv2.destroyAllWindows()