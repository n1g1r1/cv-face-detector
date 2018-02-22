#######################################################
# Face detection script
# @author: Christian Reichel
# Based on: OpenCV tutorial for face detection
# Version: 0.1
# -----------------------------------------------------
# This script detects well aligned faces and eyes with 
# haar cascades and draws boundary boxes.
# 
# Knowm challenges:
# [ ] Detect tilted faces.
# [ ] Detect eyes with glasses on.
# [ ] Cut out eyes and show as separate windows.
#######################################################

# IMPORTS
import numpy as np
import cv2 as cv

# Pretrained haar cascade classifiers.
face_cascade = cv.CascadeClassifier('data/cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('data/cascades/haarcascade_eye.xml')

# Some settings and parameters.
boundary_box_thickness = 2

# Get camera image.
camera = cv.VideoCapture(0)
output, camera_image = camera.read()

while output and cv.waitKey(1) == -1:    
    
    # Resize camera image for faster recognition.
    # Without blurring - we don't need undersampling handling at the moment.
    camera_image = cv.resize(camera_image, (0,0), fx = 0.3, fy = 0.3)

    # Save grayscale image.
    grayscale_image = cv.cvtColor(camera_image, cv.COLOR_BGR2GRAY)

    # Get face coordinates.
    faces = face_cascade.detectMultiScale(grayscale_image, 1.3, 5)

    # Draw each face.
    for (x,y,w,h) in faces:

        # Draw bounding box.
        camera_image = cv.rectangle(camera_image, (x,y), (x+w, y+h), (255, 0, 0), boundary_box_thickness)

        # Specify region of interest for gray and color. 
        # We just want to use that region for eye detection.
        roi_gray = grayscale_image[y:y+h, x:x+w]
        roi_color = camera_image[y:y+h,x:x+w]

        # Detect eyes in grayscale image.
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:

            # Draw eye bounding box in image / color ROI.
            cv.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), boundary_box_thickness)

    # Show the result.
    cv.imshow('FaceDetector', camera_image)

    # Grab next camera image.
    output, camera_image = camera.read()

# End procedures.
camera.release()
cv.destroyAllWindows()