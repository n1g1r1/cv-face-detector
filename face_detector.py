#######################################################
# Face detection with Haar classifiers
# @author: Christian Reichel
# Based on: OpenCV tutorial for face detection
# Version: 0.1.2
# -----------------------------------------------------
# Detects faces in an image with pretrained
# classifiers by OpenCV. Detects also eyes.
# 
# Choose the classifier by the string value 
# "classifier". Options:
# - "lbp" for local binary patterns
# - "haar" for haar cascadess
#######################################################

# IMPORTS
import numpy as np
import cv2 as cv
import os               # path finding

def detect_faces(image, classifier = "lbp", resize = False, resize_factor = 0.5, detect_eyes = False, draw_bounding_boxes = False, bounding_box_thickness = 2):
    "Detects faces in an image with pretrained classifiers. You can choose between lbp and haar classifier."

    # Depending on which method is used, load the right classifier.
    if classifier is "lbp":
        face_classifier_xml = 'data/lbpcascades/lbpcascade_frontalface_improved.xml'
        eye_classifier_xml  = 'data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
    elif classifier is "haar":
        face_classifier_xml = 'data/haarcascades/haarcascade_frontalface_default.xml'
        eye_classifier_xml  = 'data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'

    # Load pretrained haar cascade classifiers.
    face_classifier = cv.CascadeClassifier(os.path.join(os.path.dirname(__file__), face_classifier_xml))
        
    # Resize camera image for faster recognition.
    if resize:
        image = cv.resize(image, (0,0), fx = resize_factor, fy = resize_factor)

    # Save grayscale image.
    image_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Get face coordinates.
    faces = face_classifier.detectMultiScale(image_grayscale, 1.3, 5)

    eyes = None

    if detect_eyes or draw_bounding_boxes is not False:

        # If we have to do with the data of each face (detecting eyes or 
        # drawing bounding boxes), we have to iterate through each of them.
        for (x,y,w,h) in faces:

            if draw_bounding_boxes:
                # Draw face bounding box.
                image = cv.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), bounding_box_thickness)

            if detect_eyes:

                eye_classifier = cv.CascadeClassifier(os.path.join(os.path.dirname(__file__), eye_classifier_xml))

                # Specify search area for fast eye detection. 
                face_image_gray = image_grayscale[y:y+h, x:x+w]

                eyes = eye_classifier.detectMultiScale(face_image_gray)

                # Draw bounding boxes of eyes.
                if draw_bounding_boxes:
                    face_image = image[y:y+h, x:x+w]

                    for (ex, ey, ew, eh) in eyes:
                        cv.rectangle(face_image, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), bounding_box_thickness)


    # Reutrn found faces
    return faces, eyes, image