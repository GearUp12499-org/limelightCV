#!/usr/bin/python3

import cv2
import detect as script

image = cv2.imread("../images/mixed/pic115.png")
llrobot = [0.0] * 8

_, _, llpython = script.runPipeline(image, llrobot)

print(llpython)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Check -->
# pic108.png
# pic113.png
# pic115.png