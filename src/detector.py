import logging
import shutil
import cv2
from PIL import Image
from src.yolov4 import YOLO, decode, filter_boxes
import src.utils as utils
from src.config import cfg
from src.functions import *
import tensorflow as tf
# comment out below line to enable tensorflow outputs
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession






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
#	model_detector	path to Tensorflow detector file (e.g., 'models/teststrips.detector')
#	model_names	path to names file (e.g., 'models/teststrips.names')
#	images		path to input image
#	output		path to output folder
#	framework	tf, tflite, trt (default: 'tf')
#	input_size	resize images to (default: 416)
#	tiny		yolo or yolo-tiny (default: False)
#	model		yolov3 or yolov4 (default: 'yolov4')
#	iou		iou threshold (default: 0.45)
#	score		score threshold (default: 0.50)
#	count		count objects within images (default: False)
#	dont_show	dont show image output (default: True)
#	info		print info on detections (default: False)
#	crop		crop detections from images (default: False)
#	ocr		perform generic OCR on detection regions (default: False)
#	plate		perform license plate recognition (default: False)

def detect_test_strip(model_detector, model_names, images, output, 
		framework='tf', input_size=416, model='yolov4', 
		max_output_size_per_class=1, max_total_size=50, 
		iou=0.45, score=0.95, 
		tiny=False, count=False, dont_show=True, info=False, crop=True, ocr=False, plate=False):
	config = ConfigProto()
	config.gpu_options.allow_growth = True
	session = InteractiveSession(config=config)
	strides, anchors, num_class, xyscale = utils.load_config(tiny, model, model_names)

	# load model
	if framework == 'tflite':
		interpreter = tf.lite.Interpreter(model_path=model_detector)
	else:
		saved_model_loaded = tf.saved_model.load(model_detector, tags=[tag_constants.SERVING])
	
	# loop through images in list and run Yolov4 model on each
	for count, image_path in enumerate(images, 1):
		original_image = cv2.imread(image_path)
		original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
		
		image_data = cv2.resize(original_image, (input_size, input_size))
		image_data = image_data / 255.
		
		# get image name by using split method
		image_name = image_path.split('/')[-1]
		image_name = image_name.split('.')[0]
		
		images_data = []
		for i in range(1):
			images_data.append(image_data)
		images_data = np.asarray(images_data).astype(np.float32)
		
		if framework == 'tflite':
			interpreter.allocate_tensors()
			input_details = interpreter.get_input_details()
			output_details = interpreter.get_output_details()
			interpreter.set_tensor(input_details[0]['index'], images_data)
			interpreter.invoke()
			pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
			if model == 'yolov3' and tiny == True:
				boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
			else:
				boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
		else:
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
		
		# hold all detection data in one variable
		pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
		
		# read in all class names from config
		class_names = utils.read_class_names(model_names)
		
		# by default allow all classes in .names file
		allowed_classes = list(class_names.values())
		
		# if crop flag is enabled, crop each detection and save it as new image
		if crop:
			crop_path = os.path.join(output + '.crop')#, image_name)
			try:
				os.makedirs(crop_path)
			except FileExistsError:
				pass
			crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path, allowed_classes, class_names)
		
		# if ocr flag is enabled, perform general text extraction using Tesseract OCR on object detection bounding box
		if ocr:
			ocr(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, class_names)
		
		# if count flag is enabled, perform counting of objects
		if count:
			# count objects found
			counted_classes = count_objects(pred_bbox, allowed_classes, class_names, by_class = False)
			# loop through dict and print
			for key, value in counted_classes.items():
				print("Number of {}s: {}".format(key, value))
			image = utils.draw_bbox(original_image, pred_bbox, allowed_classes, class_names, info, counted_classes, read_plate = plate)
		else:
			image = utils.draw_bbox(original_image, pred_bbox, allowed_classes, class_names, info, read_plate = plate)
		
		image = Image.fromarray(image.astype(np.uint8))
		if not dont_show:
			image.show()
		image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
		cv2.imwrite(output + '.detection' + str(count) + '.png', image)
	
	# Close interactive session
	session.close()



