import numpy as np
import cv2

#Red, Yellow, Blue [1, 1, 1] means find all colors
# runPipeline() is called every frame by Limelight's backend.
def runPipeline(image, llrobot):
    # Initialize variables
    largestContour = np.array([[]])
    llpython = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    x, y, w, h = 0, 0, 0, 0
    # llrobot = [1.0, 1.0, 1.0]

    # Convert the image to HSV color space
    hsv_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply slight Gaussian blur for noise reduction
    # blurred_image = cv2.GaussianBlur(hsv_image1, (5, 5), 0)
    blurred_image = hsv_image1

    # MASSSSKKKKK 
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    # blurred_image = cv2.GaussianBlur(mask, (5, 5), 0)
    
    if llrobot[0] > 0.5:
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        red_mask1 = cv2.inRange(blurred_image, lower_red, upper_red)
        lower_red = np.array([160, 100, 100])
        upper_red = np.array([179, 255, 255])
        red_mask2 = cv2.inRange(blurred_image, lower_red, upper_red)
        red_mask = red_mask1 + red_mask2
        mask += red_mask
    if llrobot[1] > 0.5:
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(blurred_image, lower_yellow, upper_yellow)
        mask += yellow_mask
    if llrobot[2] > 0.5:
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(blurred_image, lower_blue, upper_blue)
        mask += blue_mask

    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # find contours in the new binary image
    contours, _ = cv2.findContours(mask, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if contours have been detected, draw them 
    if len(contours) > 0:
        # Find the largest contour
        largestContour = max(contours, key=cv2.contourArea)

        # Fit a minimum area rectangle around the largest contour
        rect = cv2.minAreaRect(largestContour)
        box = cv2.boxPoints(rect)  # Get the four corners of the rectangle
        box = np.intp(box)
        x, y = rect[0] # center of the sample
        w, h = rect[1]
        angle = rect[2]
        if w > h:
            w, h = h, w
            angle += 90
        ta = w * h
        # Check if the point is inside the rotated rect
        camMidpoint = (310, 250)
        #Step 1: translate the rect so the center is (0, 0)
        point = (camMidpoint[0] - x, camMidpoint[1] - y)

        # Step 2: rotate everything by -angle around (0, 0)
        theta = np.deg2rad(-angle)
        point = (
            point[0] * np.cos(theta) - point[1] * np.sin(theta),
            point[0] * np.sin(theta) + point[1] * np.cos(theta)
        )
        xprime = point[0]
        yprime = point[1]
        height = 300
        width = 140

        # Step 3: check if the point is within the bounds of the non-rotated rect
        inside = 0.0
        if(xprime > width / 2 or xprime < -width / 2 or yprime > height / 2 or yprime < -height / 2):
            inside = 0.0
        else:
            inside = 1.0
        
        ratio = h/w
        if ratio < 1.25 or ratio > 2.5:
            inside = 0
        if w < 100:
            inside = 0
        
        # record some custom data to send back to the robot
        llpython = [1, x, y, w, h, ta, angle, 0, inside]
        # Draw the contour and minimum area rectangle
        cv2.drawContours(image, [largestContour], 0, (0, 255, 0), 2)
        # cv2.drawContours(image, [box], 0, (0, 255, 255), 2)
        
        cv2.putText(image, f"Angle: {angle:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, f"W: {w:.2f}", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(image, f"H: {h:.2f}", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(image, f"X: {x:.2f}", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(image, f"Y: {y:.2f}", (60, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(image, f"X': {xprime}", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(image, f"Y': {yprime}", (60, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(image, f"inside: {inside}", (60, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        
    # return the largest contour for the LL crosshair, the modified image, and custom robot data
    return largestContour, image, llpython  