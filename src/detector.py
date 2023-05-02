import os
import logging
import shutil
import cv2
from PIL import Image
import numpy as np
from src.config import cfg
import src.utils as utils
from src.yolov4 import YOLO, decode, filter_boxes

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# comment out below line to enable tensorflow outputs
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



# Convert YOLO detector to Tensorflow detector
#
# Paramaters:
#	model_weights		path to weights file (e.g., 'models/teststrips.weights')
#	model_names		path to names file (e.g., 'models/teststrips.names')
#	output			path to output (e.g., 'models/teststrips.yolov4-416')
#	outdir_overwrite	overwrite output director directory (default: True)
#	tiny			is yolo-tiny or not (default: False)
#	input_size		define input size of export model (default: 416)
#	score_thres		define score threshold (default: 0.2)
#	framework		define what framework do you want to convert (tf, trt, tflite) (default: 'tf')
#	model			yolov3 or yolov4 (default: 'yolov4')
def convert_detector(model_weights, model_names, output, outdir_overwrite=True, 
		tiny=False, input_size = 416, score_thres = 0.2, framework = 'tf', model='yolov4'):
	logging.info('Converting %s into Tensorflow model (output into %s)', model_weights, output) ## INFO
	
	## Check if weigths and names files exist
	EXIT = False
	if not os.path.exists(model_weights):
		EXIT = True
		logging.error('Weights file (%s) does not exist!', model_weights) ## ERROR
	if not os.path.exists(model_names):
		EXIT = True
		logging.error('Names file (%s) does not exist!', model_names) ## ERROR
	if EXIT:
		sys.exit(1)
	
	## Remove exisiting model directory if it exists
	if os.path.exists(output):
		if outdir_overwrite:
			logging.info('Detector directory %s already exists (from a previous run?) removing it so we can recreate it', output) ## INFO
			shutil.rmtree(output)
		else:
			logging.error('Detector directory %s already exists (from a previous run?), will not overwrite!') ## ERROR
			sys.exit(1)
	
	## Load veriables
	strides, anchors, num_class, xyscale = utils.load_config(tiny, model, model_names)
	input_layer = tf.keras.layers.Input([input_size, input_size, 3])
	feature_maps = YOLO(input_layer, num_class, model, tiny)
	bbox_tensors = []
	prob_tensors = []
	
	if tiny:
		for i, fm in enumerate(feature_maps):
			if i == 0:
				output_tensors = decode(fm, input_size // 16, num_class, strides, anchors, i, xyscale, framework)
			else:
				output_tensors = decode(fm, input_size // 32, num_class, strides, anchors, i, xyscale, framework)
			bbox_tensors.append(output_tensors[0])
			prob_tensors.append(output_tensors[1])
	else:
		for i, fm in enumerate(feature_maps):
			if i == 0:
				output_tensors = decode(fm, input_size // 8, num_class, strides, anchors, i, xyscale, framework)
			elif i == 1:
				output_tensors = decode(fm, input_size // 16, num_class, strides, anchors, i, xyscale, framework)
			else:
				output_tensors = decode(fm, input_size // 32, num_class, strides, anchors, i, xyscale, framework)
			bbox_tensors.append(output_tensors[0])
			prob_tensors.append(output_tensors[1])
	pred_bbox = tf.concat(bbox_tensors, axis=1)
	pred_prob = tf.concat(prob_tensors, axis=1)
	
	if framework == 'tflite':
		pred = (pred_bbox, pred_prob)
	else:
		boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=score_thres, input_shape=tf.constant([input_size, input_size]))
		pred = tf.concat([boxes, pred_conf], axis=-1)
	
	model = tf.keras.Model(input_layer, pred)
	utils.load_weights(model, model_weights, model, tiny)
	model.summary()
	model.save(output)
	
	logging.info('Done converting model') ## INFO



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


