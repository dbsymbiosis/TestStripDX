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

def detect_test_strip(image_path, output, intervals,
		input_size=416):
	
	original_image = cv2.imread(image_path)
	original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
	
	image_data = cv2.resize(original_image, (input_size, input_size))
	image_data = image_data / 255.
	
	# hold all detection data in one variable
	bboxes = np.array([ [xmin, ymin, xmax, ymax] for name, time, xmin, xmax, ymin, ymax in intervals], dtype=np.float32)
	times = np.array([ [time] for name, time, xmin, xmax, ymin, ymax in intervals], dtype=np.int32)
	names = np.array([ [name] for name, time, xmin, xmax, ymin, ymax in intervals], dtype=np.str)
	num_objects = len(names)
	pred_bbox = [bboxes, names, times, num_objects]
	
	# crop each detection and save it as new image
	crop_path = os.path.join(output + '.crop')#, image_name)
	try:
		os.makedirs(crop_path)
	except FileExistsError:
		pass
	crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path)
	
	# draw colored boxes on image
	image = utils.draw_bbox(original_image, pred_bbox)
	
	image = Image.fromarray(image.astype(np.uint8))
	image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
	cv2.imwrite(output + '.detection' + str(count) + '.png', image)
	
	# Close interactive session
	session.close()


