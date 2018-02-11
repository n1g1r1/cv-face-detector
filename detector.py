# Detector file

import cv2 as cv
import numpy as np

# Get camera image
cameraCapture = cv.VideoCapture(0)

# Write an event listener for mouse input
clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv.EVENT_LBUTTONDOWN:
        clicked = True

# TODO: Make some precomputations for better detection



# TODO: Detect face

# TODO: Recognise face

# If not recognized => learn new face
# If recognized => write name

# Display camera frames in a window
cv.namedWindow('FaceDetector')
cv.setMouseCallback('FaceDetector', onMouse)

print ('Showing Camera feed. Click window to stop.')

success, frame = cameraCapture.read()
while success and cv.waitKey(1) == -1 and not clicked:
    cv.imshow('FaceDetector', frame)
    success, frame = cameraCapture.read()

cv.destroyWindow('FaceDetector')