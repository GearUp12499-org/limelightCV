import cv2
import numpy as np

# global constants go here
SAMPLE_DIMENSIONS = {
    "x": 3.9,  # width, in cm
    "y": 8.9,  # height, in cm
    "z": 3.9,  # depth, in cm
}

FOV = {
    "x": 54.5,  # horizontal field of view, in degrees
    "y": 42.0,  # vertical field of view, in degrees
}

# centimeter to pixel formula
pixel = lambda cm: cm * 37.8

# calculate distance to target
def calculateDistance(w, h) -> float:
    # focal length in the x and y direction
    focal_length_x = pixel(SAMPLE_DIMENSIONS["x"]) / (2 * np.tan(np.radians(FOV["x"] / 2)))
    focal_length_y = pixel(SAMPLE_DIMENSIONS["y"]) / (2 * np.tan(np.radians(FOV["y"] / 2)))
    
    # calculate distance based on width and height
    distance_x = (SAMPLE_DIMENSIONS["x"] * focal_length_x) / w
    distance_y = (SAMPLE_DIMENSIONS["y"] * focal_length_y) / h
    
    distance = (distance_x + distance_y) / 2
    
    return distance

# To change a global variable inside a function,
# re-declare it with the 'global' keyword
def incrementTestVar():
    global testVar
    testVar = testVar + 1
    if testVar == 100:
        print("test")
    if testVar >= 200:
        print("print")
        testVar = 0

def drawDecorations(image):
    cv2.putText(
        image, 
        'Limelight python script!', 
        (0, 230), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        .5, (0, 255, 0), 1, cv2.LINE_AA
    )

# runPipeline() is called every frame by Limelight's backend.
def runPipeline(image, llrobot):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # detect red.
    # Red is the union of two ranges in HSV space
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    red = mask0 + mask1

    # Reduce noise
    kernel = np.ones((7, 7), np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(red, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largestContour = np.array([[]])
    llpython = [0, 0, 0, 0, 0, 0, 0, 0]

    if len(contours) > 0:
        cv2.drawContours(image, contours, -1, 255, 2)
        largestContour = max(contours, key=cv2.contourArea)
        
        if largestContour.size > 0:  # check if the largest contour is valid
            x, y, w, h = cv2.boundingRect(largestContour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            llpython = [1, x, y, w, h, 9, 8, 7]

            # calculate distance based on width and height of the bounding box
            distance = calculateDistance(w, h)
            print(f"Distance to target: {distance} cm")
    
    incrementTestVar()
    drawDecorations(image)

    # make sure to return a contour,
    # an image to stream,
    # and optionally an array of up to 8 values for the "llpython"
    # networktables array
    return largestContour, image, llpython
