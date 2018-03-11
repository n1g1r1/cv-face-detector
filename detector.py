# IMPORTS
import numpy as np
import cv2 as cv
import os               # path finding
import time

def detect(image, classifier = 'lbp', resize = False, resize_factor = 0.5, detect_eyes = False, draw_bounding_boxes = False, bounding_box_thickness = 2, save_faces = False, face_path = 'data/training', filename = None):
    '''
    Detects faces in an image with pretrained classifiers. You can choose between lbp and haar classifier.
    '''

    # Depending on which method is used, load the right classifier.
    if classifier is 'lbp':
        face_classifier_xml = 'data/lbpcascades/lbpcascade_frontalface_improved.xml'
        eye_classifier_xml  = 'data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
    else:
        face_classifier_xml = 'data/haarcascades/haarcascade_frontalface_default.xml'
        eye_classifier_xml  = 'data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'

    # Load pretrained classifier.
    face_classifier = cv.CascadeClassifier(os.path.join(os.path.dirname(__file__), face_classifier_xml))

    # Resize camera image for faster recognition.
    if resize:
        image = cv.resize(image, (0,0), fx = resize_factor, fy = resize_factor)

    # Save grayscale image.
    image_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Get face coordinates.
    faces = face_classifier.detectMultiScale(image_grayscale, 1.3, 5)

    eyes = None

    if (detect_eyes or save_faces or draw_bounding_boxes) and len(faces) > 0:

        # If we have to do with the data of each face (detecting eyes or
        # drawing bounding boxes), we have to iterate through each of them.
        for (x,y,w,h) in faces:

            # Specify search area for fast eye detection.
            face_image = image[y:y+h, x:x+w]
            face_image_gray = image_grayscale[y:y+h, x:x+w]

            if save_faces:
                # Set filename.
                filename = filename + "." + str(hash(time.time())) + '.jpg'
                print('Save face as ' + filename)
                try:
                    cv.imwrite(face_path + '/' + filename, face_image)
                except cv.error as e:
                    print(e)

            if draw_bounding_boxes:
                # Draw face bounding box.
                image = cv.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), bounding_box_thickness)

            if detect_eyes:

                eye_classifier = cv.CascadeClassifier(os.path.join(os.path.dirname(__file__), eye_classifier_xml))

                eyes = eye_classifier.detectMultiScale(face_image_gray)

                if draw_bounding_boxes:
                    for (ex, ey, ew, eh) in eyes:
                        cv.rectangle(face_image, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), bounding_box_thickness)

    # Reutrn found faces
    return faces, eyes, image
