import os
import logging
import shutil
import cv2
from PIL import Image
import numpy as np
import src.utils as utils
from ultralytics import YOLO



# Run Tensorflow test strip detector on provided image file
#
# Parameters:
#	model_detector_path	path to Tensorflow detector file (e.g., 'models/teststrips.detector')
#	model_names_path	path to names file (e.g., 'models/teststrips.names')
#	model_names		names of model objects
#	image_data		input image data
#	iou			iou threshold (default: 0.45)
#	score			score threshold (default: 0.50)
#	input_size		resize images to (default: 416)
def detect_test_strip(model_detector_path, model_names_path, model_names, 
		original_image, input_size=416,
		max_output_size_per_class=1, max_total_size=50, 
		iou=0.45, score=0.95):
	
	# config
	strides, anchors, num_class, xyscale = utils.load_config(False, 'yolov4', model_names_path)
	
	image_data = cv2.resize(original_image, (input_size, input_size))
	image_data = image_data / 255.
	
	images_data = []
	for i in range(1):
		images_data.append(image_data)
	images_data = np.asarray(images_data).astype(np.float32)
	
	# load model
	saved_model_loaded = tf.saved_model.load(model_detector_path, tags=[tag_constants.SERVING])
	
	infer = saved_model_loaded.signatures['serving_default']
	batch_data = tf.constant(images_data)
	pred_bbox = infer(batch_data)
	for key, value in pred_bbox.items():
		boxes = value[:, :, 0:4]
		pred_conf = value[:, :, 4:]
	
	# run non max suppression on detections
	boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
		boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
		scores=tf.reshape(
			pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
		max_output_size_per_class=max_output_size_per_class,
		max_total_size=max_total_size,
		iou_threshold=iou,
		score_threshold=score
	)
	
	# format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
	original_h, original_w, _ = original_image.shape
	bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
	
	# reformat from tf to numpy objects
	scores = scores.numpy()[0]
	classes = classes.numpy()[0]
	valid_detections = valid_detections.numpy()[0]
	
	# Get just valid detections
	bboxes  = [bboxes[x]  for x in range(valid_detections)]
	scores  = [scores[x]  for x in range(valid_detections)]
	classes = [classes[x] for x in range(valid_detections)]
	names   = [model_names[int(x)] for x in classes]
	
	# hold all detection data in one variable
	# data: bboxes, scores, names, num_objects
	pred_bbox = {"bboxes":bboxes, "scores":scores, "names":names, "num_objects":valid_detections}
	
	return(pred_bbox)


