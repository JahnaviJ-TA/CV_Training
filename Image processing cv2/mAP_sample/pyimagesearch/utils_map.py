# import the necessary packages
from pyimagesearch import config
from progressbar import progressbar
from darknet import darknet
import pickle
import json
import cv2
import os
def image_detection(imagePath, network, classNames, thresh):
	# image_detection takes the image as an input and returns
	# the detections to the calling function
	width = darknet.network_width(network)
	height = darknet.network_height(network)
	
	# create an empty darknetImage of shape [608, 608, 3]
	darknetImage = darknet.make_image(width, height, 3)
	
  # read the image from imagePath, convert from BGR->RGB
	# resize the image as per YOLOv4 network input size
	image = cv2.imread(imagePath)
	imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	imageResized = cv2.resize(imageRGB, (width, height),
		interpolation=cv2.INTER_LINEAR)
	darknet.copy_image_from_bytes(darknetImage, imageResized.tobytes())
	
	# detections include the class labels, bounding box coordinates, and
	# confidence score for all the proposed ROIs in the image
	detections = darknet.detect_image(network, classNames, darknetImage,
		thresh=thresh)
	darknet.free_image(darknetImage)
	
	# return the detections and image shape to the calling function
	return (detections, image.shape)