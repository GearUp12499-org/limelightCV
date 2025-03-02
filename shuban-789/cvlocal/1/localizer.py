#!/usr/bin/python3

import script
import cv2

category = "mixed"
image = cv2.imread("../images/" + category + "/" + "pic110.png")
result = script.runPipeline(image, [0.0, 1.0, 0.0])
cv2.imshow("Result", result[1])
print("X:", result[2][1])
print("Y:", result[2][2])
print("w:", result[2][3])
print("h:", result[2][4])
print("angle:", result[2][6])
print("inside:", result[2][8])
cv2.waitKey(0)
cv2.destroyAllWindows()  