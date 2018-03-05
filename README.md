# CV module: Face detection 
A simple face detection module for Python and OpenCV projects. Currently supported classifiers:

- LBP (Local Binary Pattens) as default.
- Haar cascade.

If needed, the detector also returns the image with the drawn boundary boxes on it.

## How to use the detector

Call the detector method by the following call:

```
faces, eyes, image = detect_faces(YOUR_IMAGE)
```

- `YOUR_IMAGE`: your image as pre-read numpy array (with `image = cv2.imread(YOUR_IMAGE)`).

### Return values

You get an array with face and - if you choose so - eye positions. Also, you get back the image itself which might have bounding boxes, if you have chosen to `draw_bounding_boxes`.

### Documentation

```
faces, eyes, image = detect_faces(YOUR_IMAGE, classifier = "lbp", resize = False, resize_factor = 0.5, detect_eyes = False, draw_bounding_box = False, bounding_box_thickness = 2)
```

- `classifier`: the chosen classifier. You can choose `lbp` for Local Binary Pattern or `haar` for Haar Classifiers. Default: `lbp`.
- `resize`: resize the image for faster recognition. Default: `False`.
- `resize_factor`: specifies the factor of resizing the image. Default: `0.5`
- `detect_eyes`: `True` or `False` if you want to detect also eyes. Default: `False`.
- `draw_bounding_box`: Option to draw bounding boxes around faces - and if `detect_eyes` is `True` - also around eyes. Default: `False`
- `bounding_box_thickness`: specifies the thickness of the bounding box. Default: `2`.