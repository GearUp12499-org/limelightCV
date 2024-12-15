import cv2
import numpy as np

# global constants go here:
SAMPLE_DIMENSIONS = {
    "x": 3.9, # width, in cm
    "y": 8.9, # height, in cm
    "z": 3.9, # depth, in cm
}

FOV = {
    "x": 59.0, # horizontal field of view, in degrees
    "y": 49.7, # vertical field of view, in degrees
}

# global variables go here:
testVar = 0

# cm to pixels equation
cm2pix = lambda cm: round(cm * 37.8)

# To change a global variable inside a function,
# re-declare it with the 'global' keyword

def calculateDistance(Ta) -> float:
    # calculate distance to target in cm
    # note from shuban: AI gave me this formula, I want to see if it works
    return (SAMPLE_DIMENSIONS["x"] * cm2pix(Ta)) / (2 * cm2pix(SAMPLE_DIMENSIONS["y"] * np.tan(np.radians(FOV["y"] / 2))))

def incrementTestVar():
    global testVar
    testVar = testVar + 1
    if testVar == 100:
        print("test")
    if testVar >= 200:
        print("print")
        testVar = 0

def drawDecorations(image):
    cv2.putText(image, 
        'Limelight python script!', 
        (0, 230), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        .5, (0, 255, 0), 1, cv2.LINE_AA)
    
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
    red = mask0+mask1

    # Reduce noise
    kernel = np.ones((7,7), np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
    

    contours, _ = cv2.findContours(red, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
    largestContour = np.array([[]])
    llpython = [0,0,0,0,0,0,0,0]

    if len(contours) > 0:
        cv2.drawContours(image, contours, -1, 255, 2)
        largestContour = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(largestContour)

        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
        llpython = [1,x,y,w,h,9,8,7]  
  
    Ta = cv2.contourArea(largestContour)
    print(f"Ta: {Ta}")
    print(f"Distance to target: {calculateDistance(Ta)}")
    incrementTestVar()
    drawDecorations(image)
    
    # make sure to return a contour,
    # an image to stream,
    # and optionally an array of up to 8 values for the "llpython"
    # networktables array
    return largestContour, image, llpython