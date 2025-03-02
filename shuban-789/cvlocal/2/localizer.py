#!/usr/bin/python3

import cv2
import script

image = cv2.imread("../images/mixed/pic110.png")
llrobot = [0.0] * 8

_, _, llpython = script.runPipeline(image, llrobot)

print(llpython)

cv2.waitKey(0)
cv2.destroyAllWindows()