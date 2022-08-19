#!/usr/bin/env python3
DESCRIPTION = '''
TestStripDX

An image processing framework for processing and extracting test strip results from a photo.
'''
import sys
import os
import argparse
import logging
import shutil

import numpy as np
from moviepy import *
import moviepy.editor as mpy
from PIL import Image
import imageio.v2 as imageio

#### convert_detector
import tensorflow as tf
from src.yolov4 import YOLO, decode, filter_boxes
import src.utils as utils
from src.config import cfg

#### process_video
# comment out below line to enable tensorflow outputs
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import src.utils as utils
from src.functions import *
from tensorflow.python.saved_model import tag_constants
import cv2
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


# TODO: Fix needing to specify names file in src/config.py

## Pass arguments.
def main():
	## Pass command line arguments. 
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=DESCRIPTION)
	subparsers = parser.add_subparsers(dest='command', required=True)
	
	# Parser for the conversion of the yolov4 to Tensorflow detector
	parser_convert_detector = subparsers.add_parser('convert', 
		help='Convert yolov4 detector to Tensorflow detector',
		description=CONVERT_RESULTS_DESCRIPTION
	)
	parser_convert_detector.add_argument('-m', '--model', metavar='model_name', 
		required=True, type=str, 
		help='Name of model in models/ directory to convert (NOTE: models/model_name.weights file needs to exist)'
	)
	parser_convert_detector.add_argument('--debug',
		required=False, action='store_true',
		help='Print DEBUG info (default: %(default)s)'
	)
	
	# Parser for the processing of the test strip video files
	parser_process_video = subparsers.add_parser('process', help='Process test strip video files')
	parser_process_video.add_argument('-v', '--videos', metavar='teststrip.mp4', 
		required=True, nargs='+', type=str, 
		help='Video files to process'
	)
	parser_process_video.add_argument('-m', '--model', metavar='model_name',
		required=False, type=str, default='URS10',
		help='Name of model in models/ directory to convert (NOTE: models/model_name.weights file needs to exist)'
	)
	parser_process_video.add_argument('-s', '--suffix', metavar='TestStripDX',
		required=False, type=str, default='.TestStripDX',
		help='Prefix to add to TestStripDX output files (default: %(default)s)'
	)
	parser_process_video.add_argument('-c', '--cleanup',
		required=False, action='store_true',
		help='Remove detection images from Tensorflow (default: %(default)s)'
	)
	parser_process_video.add_argument('--debug',
		required=False, action='store_true',
		help='Print DEBUG info (default: %(default)s)'
	)
	
	# Parser for the combining of the results files into a single output
	parser_combine_results = subparsers.add_parser('combine', 
		help='Combine results from processed video files', 
		description=COMBINE_RESULTS_DESCRIPTION
	)
	parser_combine_results.add_argument('-t', '--test_results', metavar='test_results.txt',
		required=True, nargs='+', type=argparse.FileType('r'),
		help='Input test strip results files (required)'
	)
	parser_combine_results.add_argument('-o', '--out', metavar='output.txt',
		required=False, default=sys.stdout, type=argparse.FileType('w'),
		help='Output file (default: stdout)'
	)
	parser_combine_results.add_argument('-b', '--blank_results', metavar='blank_results.txt',
		required=False, default=None, type=argparse.FileType('r'),
		help='Input blank strip results files (default: not used)'
	)
	parser_combine_results.add_argument('--debug',
		required=False, action='store_true',
		help='Print DEBUG info (default: %(default)s)'
	)
	
	# Parse all arguments.
	args = parser.parse_args()
	
	## Set up basic debugger
	logFormat = "[%(levelname)s]: %(message)s"
	logging.basicConfig(format=logFormat, stream=sys.stderr, level=logging.INFO)
	if args.debug:
		logging.getLogger().setLevel(logging.DEBUG)
	
	logging.debug('%s', args) ## DEBUG
	
	models_dir = 'models'
	
	if args.command == 'convert':
		convert_detector(os.path.join(models_dir, args.model+'.weights'),
			os.path.join(models_dir, args.model+'.names'), 
			os.path.join(models_dir, args.model+'.detector')
		)
	elif args.command == 'process':
		process_video(args.videos, 
			os.path.join(models_dir, args.model+'.detector'), 
			os.path.join(models_dir, args.model+'.names'), 
			(('Glucose', 30), ('Ketone', 40), ('Blood', 60), ('Leukocytes', 120)),
			args.cleanup,
			args.suffix
		)
	elif args.command == 'combine':
		combine_results(args.test_results, args.blank_results, args.out, ["Leukocytes", "Glucose", "Ketone", "Blood"])





#####################
#####################
## Conver detector ##
#####################
#####################
CONVERT_RESULTS_DESCRIPTION = '''
Convert yolov4 detector to Tensorflow detector

Needs models/model_name.weights and models/model_name.names to exist.
'''
# Paramaters:
#	weights		path to weights file (default: 'models/teststrips.weights')
#	names		path to names file (default: 'models/teststrips.names')
#	output		path to output (default: 'models/teststrips.yolov4-416')
#	tiny		is yolo-tiny or not (default: False)
#	input_size	define input size of export model (default: 416)
#	score_thres	define score threshold (default: 0.2)
#	framework	define what framework do you want to convert (tf, trt, tflite) (default: 'tf')
#	model		yolov3 or yolov4 (default: 'yolov4')

def convert_detector(weights, names, output, tiny=False, input_size = 416, score_thres = 0.2, framework = 'tf', model='yolov4', outdir_overwrite=True):
	logging.info('Converting %s into Tensorflow model (output into %s)', weights, output) ## INFO
	
	# Check if weigths file exist
	if not os.path.exists(weights):
		logging.error('Weights file (%s) does not exist!', weights) ## ERROR
		sys.exit(1)
	
	# Remove exisiting model directory if it exists
	if os.path.exists(output):
		if outdir_overwrite:
			logging.info('Detector directory %s already exists (from a previous run?) removing it so we can recreate it', output) ## INFO
			shutil.rmtree(output)
		else:
			logging.error('Detector directory %s already exists (from a previous run?), will not overwrite!') ## ERROR
			sys.exit(1)
	
	# Previously load_config() from src/utils.py
	if tiny:
		strides = np.array(cfg.YOLO.STRIDES_TINY)
		anchors = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, tiny)
		xyscale = cfg.YOLO.XYSCALE_TINY if model == 'yolov4' else [1, 1]
	else:
		strides = np.array(cfg.YOLO.STRIDES)
		if model == 'yolov4':
			anchors = utils.get_anchors(cfg.YOLO.ANCHORS, tiny)
		elif model == 'yolov3':
			anchors = utils.get_anchors(cfg.YOLO.ANCHORS_V3, tiny)
		xyscale = cfg.YOLO.XYSCALE if model == 'yolov4' else [1, 1, 1]
	num_class = len(utils.read_class_names(names))
	# End load_config()
	
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
	utils.load_weights(model, weights, model, tiny)
	model.summary()
	model.save(output)
	logging.info('Done converting model') ## INFO





####################
####################
## Process videos ##
####################
####################

# Parameters:
#	videos	videos to process
#	
def process_video(videos, detector, detector_names, intervals, cleanup, outdir_suffix, outdir_overwrite=True):
	logging.info('Processing video files') ## INFO
	
	# Extract names and times to extract from video.
	names = [x[0] for x in intervals]
	times = [x[1] for x in intervals]
	
	for video in videos:
		outdir = video+outdir_suffix
		results_file = outdir+'.results.txt'
		frame_prefix = os.path.join(outdir, "frame")
		detection_images = []
		detection_pdf_path = outdir+'.detection.pdf'
		
		logging.info('########################################################') ## INFO
		logging.info('## Extracting frames from %s', video) ## INFO
		logging.info('########################################################') ## INFO
		
		# Create output directory
		logging.debug('out_dir=%s', outdir) ## DEBUG
		
		# Check if video file exists.
		if not os.path.exists(video):
			logging.error('Video file %s does not exists!', video) ## ERROR
			sys.exit(1)
		
		# Remove exisiting model directory if it exists
		if os.path.exists(outdir):
			if outdir_overwrite:
				logging.info('Output directory %s already exists (from a previous run?), removing it so we can recreate it', outdir) ## INFO
				shutil.rmtree(outdir)
			else:
				logging.error('Output directory %s already exists (from a previous run?), will not overwrite!') ## ERROR
				sys.exit(1)
		
		# Create output directory
		os.mkdir(outdir)
		
		# Open results file
		results = open(results_file, 'w')
		
		# Extract frame from a specific timestamp in a video.
		capture_frame(video, frame_prefix, times)
		
		# Extract tests from frames
		for name, time in intervals:
			frame_in = frame_prefix+"."+str(time)+"sec.png"
			frame_out = frame_prefix+"."+str(time)+"sec.detect"
			detection_images.append(frame_prefix+"."+str(time)+"sec.detect.detection1.png")
			
			logging.info('Searching %s frame at time %s seconds', name, time) ## INFO
			logging.debug('In frame: %s', frame_in) ## DEBUG
			logging.debug('Out frame: %s', frame_out) ## DEBUG
			detect_tests(detector, detector_names, [frame_in], frame_out)
			
			# Check to see if our target frame has been extracted
			target_frame = os.path.join(frame_prefix+"."+str(time)+"sec.detect.crop", name+"_1.png")
			logging.debug('Frame: %s', target_frame) ## DEBUG
			if os.path.exists(target_frame):
				score = extract_colors(target_frame)
			else:
				logging.warning('No frame detected for %s frame at time %s seconds', name, time) ## WARNING
				score = 'NA'
			logging.debug('Score: %s', score) ## DEBUG
			
			# Write score info to file
			results.write(name+'\t'+str(score)+'\n')
			
			# Check to see if multiple frames have been detected - give warning if found
			target_frame = os.path.join(frame_prefix+"."+str(time)+"sec.detect", name+"_2.png")
			logging.debug('Frame: %s', target_frame) ## DEBUG
			if os.path.exists(target_frame):
				logging.warning('Multiple test regions identified for %s frame at time %s seconds', name, time) ## WARNING
		
		# Close results file
		results.close()
		
		# Create combined detection image pdf
		logging.debug('detection_images: %s', detection_images) ## DEBUG
		Image.open(detection_images[0]).save(detection_pdf_path, "PDF", resolution=100.0, save_all=True, 
			append_images=(Image.open(f) for f in detection_images[1:])
		)
		
		# Cleanup if required
		if cleanup:
			logging.info('Cleaning up - removing %s', outdir) ## INFO
			shutil.rmtree(outdir)
		
		logging.info('########################################################') ## INFO
		logging.info('## Finished. Results in %s', results_file) ## INFO
		logging.info('########################################################') ## INFO
	
	logging.info('Finished processing video files') ## INFO



# Parameters:
#	video_filename	Input video files
#	out_prefix	Pefix to use for frames that we extract from video
#	seconds		Second into video to grab frame from
def capture_frame(video_filename, out_prefix, seconds):
	seconds = sorted(seconds) # sort so it is ordered smallest to largest
	vid = mpy.VideoFileClip(video_filename)
	
	# Check that seconds argument does not excede total video duration
	logging.debug('Video rotation: %s', vid.rotation) ## DEBUG
	logging.debug('Video duration: %s seconds', vid.duration) ## DEBUG
	if seconds[-1] > vid.duration:
		logging.error('The length (%s seconds) is shorter then the number of seconds that we want a frame from (%s seconds)!', vid.duration, seconds[-1])
		exit(1)
	
	# Extract first frame with timestamp higher then what is requested. 
	vid = video_rotation(vid)
	for i, (tstamp, frame) in enumerate(vid.iter_frames(with_times=True)):
		if tstamp > seconds[0]:
			logging.info('Found frame for %s seconds: frame_count:%s; timestamp:%s', seconds[0], i, tstamp) ## DEBUG
			img = Image.fromarray(frame, 'RGB')
			frame_filename = out_prefix + '.' + str(seconds[0]) + 'sec.png'
			img.save(frame_filename)
			seconds = seconds[1:] # Remove first element from list as we just found a frame for this timepoint
		# Break loop if we have run out of timepoints that we want.
		if len(seconds) == 0:
			logging.info("Done extracting frames from video")
			break



def video_rotation(video):
	"""
	   Rotate video back to 0
	   Code based on https://github.com/Zulko/moviepy/issues/586
	"""
	if video.rotation in (90, 270):
		video = video.resize(video.size[::-1])
		video.rotation = 0
	else:
		video = video.rotate(90)
	return video


def extract_colors(image_filename):
	pic = imageio.imread(image_filename)
	R = pic[ :, :, 0]
	G = pic[ :, :, 1]
	B = pic[ :, :, 2]
	meanR = np.mean(R)
	meanG = np.mean(G)
	meanB = np.mean(B)
	return ((meanR+meanG+meanB)/3)



# Parameters:
#	detector	path to Tensorflow detector file
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

def detect_tests(detector, detector_names, images, output, 
		framework='tf', input_size=416, model='yolov4', iou=0.45, score=0.50, 
		tiny=False, count=True, dont_show=True, info=True, crop=True, ocr=False, plate=False):
	config = ConfigProto()
	config.gpu_options.allow_growth = True
	session = InteractiveSession(config=config)

	# Previously load_config() from src/utils.py
	if tiny:
		strides = np.array(cfg.YOLO.STRIDES_TINY)
		anchors = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, tiny)
		xyscale = cfg.YOLO.XYSCALE_TINY if model == 'yolov4' else [1, 1]
	else:
		strides = np.array(cfg.YOLO.STRIDES)
		if model == 'yolov4':
			anchors = utils.get_anchors(cfg.YOLO.ANCHORS, tiny)
		elif model == 'yolov3':
			anchors = utils.get_anchors(cfg.YOLO.ANCHORS_V3, tiny)
		xyscale = cfg.YOLO.XYSCALE if model == 'yolov4' else [1, 1, 1]
	num_class = len(utils.read_class_names(detector_names))
	# End load_config()

	# load model
	if framework == 'tflite':
		interpreter = tf.lite.Interpreter(model_path=detector)
	else:
		saved_model_loaded = tf.saved_model.load(detector, tags=[tag_constants.SERVING])
	
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
			max_output_size_per_class=50,
			max_total_size=50,
			iou_threshold=iou,
			score_threshold=score
		)
		
		# format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
		original_h, original_w, _ = original_image.shape
		bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
		
		# hold all detection data in one variable
		pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
		
		# read in all class names from config
		class_names = utils.read_class_names(cfg.YOLO.CLASSES)
		
		# by default allow all classes in .names file
		allowed_classes = list(class_names.values())
		
		# if crop flag is enabled, crop each detection and save it as new image
		if crop:
			crop_path = os.path.join(output + '.crop')#, image_name)
			try:
				os.makedirs(crop_path)
			except FileExistsError:
				pass
			crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path, allowed_classes)
		
		# if ocr flag is enabled, perform general text extraction using Tesseract OCR on object detection bounding box
		if ocr:
			ocr(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox)
		
		# if count flag is enabled, perform counting of objects
		if count:
			# count objects found
			counted_classes = count_objects(pred_bbox, by_class = False, allowed_classes=allowed_classes)
			# loop through dict and print
			for key, value in counted_classes.items():
				print("Number of {}s: {}".format(key, value))
			image = utils.draw_bbox(original_image, pred_bbox, info, counted_classes, allowed_classes=allowed_classes, read_plate = plate)
		else:
			image = utils.draw_bbox(original_image, pred_bbox, info, allowed_classes=allowed_classes, read_plate = plate)
		
		image = Image.fromarray(image.astype(np.uint8))
		if not dont_show:
			image.show()
		image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
		cv2.imwrite(output + '.detection' + str(count) + '.png', image)
	
	# Close interactive session
	session.close()




#####################
#####################
## Combine results ##
#####################
#####################

COMBINE_RESULTS_DESCRIPTION = '''
Combine results from TestStripDX and calculate Relative Enzymatic Activity (REA) for each strip. 

If the optional blank results file is provided then REA values are calcultaed using those intensity values.
If a blank is not provided then default "blank" values of 255 will be used to calculate REA.
'''

def combine_results(test_results, blank_results, out, names):
	logging.info('Combining results files') ## INFO
	
	# Header
	t = 'Files\ttype'
	for name in names:
		t = t + '\t' + name + '_intensity\t' + name + '_REA'
	out.write(t + '\n')

	# Blank strips
	logging.debug('Loading blank results file: %s', blank_results) ## DEBUG
	blank = load_results(blank_results, names)
	t = blank_results.name + '\tblank'
	for name in names:
		t = t + '\t' + str(blank[name]) + '\tNA'
	out.write(t + '\n')

	# Test strips
	for test_file in test_results:
		logging.debug('Loading results file: %s', test_file) ## DEBUG
		results, REA = load_and_process_test_results(test_file, blank, names)
		t = test_file.name + '\ttest'
		for name in names:
			t = t + '\t' + str(results[name]) + '\t' + str(REA[name])
		out.write(t + '\n')
	
	logging.info('Done combining results files') ## INFO



def load_and_process_test_results(results_file, blank, names):
	results = load_results(results_file, names)
	REA = {}
	for name in names:
		try:
			REA[name] = blank[name] - results[name]
		except TypeError:
			REA[name] = results[name]
	return results, REA



def load_results(results_file, names):
	results = {x:255.0 for x in names}
	if results_file is not None:
		with results_file as r:
			for line in r:
				line = line.strip().split('\t')
				if line[0] in names:
					try:
						results[line[0]] = float(line[1])
					except ValueError:
						results[line[0]] = line[1]
		
		# Assume if results equals 255 assume that this name was missing from the results file.
		for name in names:
			if results[name] == 255.0:
				logging.warning('A value for %s was not found in %s results file. Setting to 255 as default.', name, results_file.name)
	logging.debug('%s', results) ## DEBUG
	return results





##########
##########
## Done ##
##########
##########

if __name__ == '__main__':
	main()
