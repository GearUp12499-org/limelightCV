import numpy as np
import cv2
import sys

# CONSTANTS
PICKUP_CENTER = ()
image = cv2.imread("../images/mixed/pic103.png")

def calculateContourPickupCenter(countours) -> tuple:
    return (0, 0)

def runPipeline(image, llrobot):
    llpython = [0.0] * 8
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_lower = np.array([20, 100, 100], dtype="uint8")
    yellow_upper = np.array([40, 255, 255], dtype="uint8")


    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    cv2.imshow('before', mask)
    mask = cv2.GaussianBlur(mask, (25,25), 0)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    target_w = 140
    target_h = 260

    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        x, y = rect[0]
        w, h = rect[1]
        angle = rect[2]

        if w > h:
            w, h = h, w
            angle += 90

        print(x, y, w, h, angle)

        ratio_w = w / target_w
        ratio_h = h / target_h

        if ratio_w < .75 or ratio_h < .75:
            continue
        
        if ratio_w > .75 and ratio_w < 1.25 and ratio_h > .75 and ratio_h < 1.25:
            cv2.drawContours(image, [box], 0, (0,255,0), 2)
            continue

        submask = np.zeros_like(mask)
        cv2.drawContours(submask, [box], -1, 1, cv2.FILLED)
        submask = submask * mask

        cv2.imshow('submask', submask)

        # Lines stuff
        edges = cv2.Canny(submask, 10, 100)
        #cv2.imshow('edges', edges)
        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(edges)[0]

        line_img = np.zeros_like(submask)
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            cv2.line(line_img, (x1, y1), (x2,y2), 255, 5)

        #cv2.imshow("lines", line_img)
        # End of lines stuff

        # sub_contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print("AAA", len(sub_contours))

        kernel = np.ones((5, 5), np.uint8)
        for _ in range(10):
            submask = cv2.erode(submask, kernel)
            sub_contours, _ = cv2.findContours(submask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            foundOne = False
            
            for c in sub_contours:
                x, y, w, h, angle = calcRectValues(c)
                rect = cv2.minAreaRect(c)
                subc_box = cv2.boxPoints(rect)
                subc_box = np.intp(subc_box)
                rw = w / target_w
                rh = h / target_h

                if rw > 0.6 and rw < 1.15 and rh > 0.6 and rh < 1.15:
                    cv2.drawContours(image, [subc_box], 0, (0,255,0), 2)
                    foundOne = True
                    continue
                
                # submask2 = cv2.erode(submask, kernel)
            # if foundOne:
            #     break
        cv2.imshow("erode", submask)
        
        # for sub_contour in sub_contours:
        #     rect = cv2.minAreaRect(sub_contour)
        #     subc_box = cv2.boxPoints(rect)
        #     subc_box = np.intp(subc_box)
            
        #     cv2.drawContours(image, [subc_box], 0, (0,255,0), 2)
            
        # allContours = contours + sub_contours
        
    cv2.imshow('original', image)
    cv2.imshow('hsv', hsv)
    #cv2.imshow('mask', mask)

    return None, image, llpython

runPipeline(image, [0.0] * 8)

def calcRectValues(contour):
    rect = cv2.minAreaRect(contour)
    x, y = rect[0]
    w, h = rect[1]
    angle = rect[2]

    if w > h:
        w, h = h, w
        angle += 90
    return x, y, w, h, angle