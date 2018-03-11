# CV module: Face Detector
A simple face detection module for Python and OpenCV projects. Currently supported classifiers:

- `lbp` [Local Binarization Pattern Histogram ](https://www.docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#local-binary-patterns-histograms).
- `haar` [Haar Cascades](https://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html).

If needed, the detector also returns the image with the drawn boundary boxes for found faces on it.

## Installation and usage

1. Open a terminal window, navigate to your project and add this python as submodule as following:

```bash
git submodule add https://github.com/n1g1r1/cv-module-face-detector modules/face_detector
```

2. Import it as python module:

```python
from modules.face_detector import detector
```

3. Call the detector:

```python
faces, eyes, image = detector.detect(image, classifier = 'lbp', resize = False, resize_factor = 0.5, detect_eyes = False, draw_bounding_boxes = False, bounding_box_thickness = 2, save_faces = False, face_path = 'data/training', filename = None)
```

## The parameters in detail

- `image`: The given image as numpy array (use `cv2.imread() or a given webcam image).
- `classifier`: The face classifier algorithm. Default: `lbp`.
- `resize`: Should the image get resized? Default: `False`.
- `resize_factor`: The resize factor, if resized. Default: `0.5`.
- `detect_eyes`: Should the algorithm also detect eyes? Default: `False`.
- `draw_bounding_boxes`: Draw bounding boxes? Default: `False`.
- `bounding_box_thickness`: Bounding box thickness. Default: `2`.
- `save_faces`: Should the algorithm save found faces? Useful to handle face capturing in one loop. Default: `False`.
- `face_path`: The path where face images should be saved. Default: `data/training`.
- `filename`: The filename. Has to be set if `save_faces` is true. Default: `None`.

## Return

```python
return faces, eyes, image
```

- `faces`: The found faces in the image as array.
- `eyes`: The found eyes in the image as array.
- `image`: The image itself. If `draw_bounding_boxes` is `True`, it returns the image with the drawn bounding boxes on it.
