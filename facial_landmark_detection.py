# pyimage search tutorial #http://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# aqui estoy haciendo que cuando corra este programa puedo pasarle el path como argumento
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
# ya esto es un modelo pre-trained para detectar lo que dice el nombre del metodo
detector = dlib.get_frontal_face_detector()
# este es el file que descargue
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image especificada en el args parametro, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
# resize it using this library made by the author https://github.com/jrosebr1/imutils
image = imutils.resize(image, width=500)
#convert it to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
#Esta funcion es de dlib line 46 y es la que detecta en el trained modelo
# le pasa la imagen como gray y el 1 significa la resolucion - pyramid layer
rects = detector(gray, 1) # detector - detect las faces en la imagen

# Given the (x, y)-coordinates of the faces in the image, we can now apply facial landmark detection to each of the face regions
# loop over the face detections i guess que es el index y rect x,y de la cara
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region (del modelo que se le pasa
    # como parametro)
	shape = predictor(gray, rect)
    # convert the facial landmark (x, y)-coordinates to a NumPy array - metodo que esta definido arrriba
	shape = face_utils.shape_to_np(shape)

    # convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box - metodo defnido arriba
	(x, y, w, h) = face_utils.rect_to_bb(rect)
    # opencv method to create rectangle
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the face number - opencv methodto put text on a image
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for (x, y) in shape:
        # draw a circle on every landmark found opencv method
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)


# show the output image with the face detections + facial landmarks
# opencv method to show an image
cv2.imshow("Output", image)
cv2.waitKey(0)
