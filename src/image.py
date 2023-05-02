import os
import logging
import shutil
import cv2
import random
import colorsys
import re
import numpy as np
from PIL import Image
import imageio as imageio
from src.detector import detect_test_strip



# Crop tests from provided image file
#
# Parameters:
#	image_path	 	path to input image
#	output		 	path to output folder
#       model_detector_path	path to Tensorflow detector file (e.g., 'models/teststrips.detector')
#       model_names_path 	path to names file (e.g., 'models/teststrips.names')
#	model_intervals	 	test intervals/coords
#	model_names	 	names of model objects
#	landmark_name		name of landmark to use for orientation
#	landmark_bounds		bounding coords of landmark
def crop_test_strip(image_path, output_path,
		model_detector_path, model_names_path, model_names, model_intervals,
		landmark_name, landmark_bounds):
	logging.info('Start cropping tests from frame: %s', image_path) ## INFO
	
	# Import and format image
	original_image = cv2.imread(image_path)
	original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
	
	# Search for landmark using ML
	logging.info(' - Using ML to search for landmark objects') ## INFO
	logging.debug('In frame: %s', image_path) ## DEBUG
	logging.debug('Out prefix: %s', output_path) ## DEBUG
	ML_pred_bbox = detect_test_strip(model_detector_path, model_names_path, model_names, original_image)
	
	# draw colored boxes on image for ML detections (used for debugging)
	ML_image = draw_bbox(original_image, ML_pred_bbox)
	ML_image = Image.fromarray(ML_image.astype(np.uint8))
	ML_image = cv2.cvtColor(np.array(ML_image), cv2.COLOR_BGR2RGB)
	cv2.imwrite(output_path + '.ML_detection.png', ML_image)
	
	# Check if we found the landmark in the image and extract it coordinates
	ML_bboxes   = ML_pred_bbox["bboxes"]
	ML_names       = ML_pred_bbox["names"]
	ML_num_objects = ML_pred_bbox["num_objects"]
	l_xmin, l_ymin, l_xmax, l_ymax = 460, 100, 530, 140
	for i in range(ML_num_objects):
		if ML_names[i] == landmark_name:
			logging.info('Landmark found') ## INFO
			l_xmin, l_ymin, l_xmax, l_ymax = ML_bboxes[i]
			break
	else:
		logging.error('Failed to find landmark in frame. Falling back to default coords, these are a rought approximation but will likely be wrong. Please double check these results.') ## ERROR
	
	# Check if landmark is where we expect - raise warning if it isnt.
	if l_xmin < landmark_bounds["xmin"] or \
	   l_xmin > landmark_bounds["xmax"] or \
	   l_ymin < landmark_bounds["ymin"] or \
	   l_ymin > landmark_bounds["ymax"]:
		logging.warning('Landmark ML (%s: xmin:%s, xmax:%s, ymin:%s, ymax:%s) was outside the expected bounds (xmin:%s, xmax:%s, ymin:%s, ymax:%s). This might mean that the video has an unexpected rotation or that the strip might not be correctly positioned in the holder.', 
			landmark_name, l_xmin, l_xmax, l_ymin, l_ymax,
			xmin, xmax, ymin, ymax) # WARNING
	
	# hold all detection data in one variable
	bboxes = np.array([  [
				l_xmin + xmin, 
				l_ymin + ymin, 
				l_xmin + xmax, 
				l_ymin + ymax
			     ] for name, time, xmin, xmax, ymin, ymax in model_intervals], dtype=np.float32)
	times = np.array([time for name, time, xmin, xmax, ymin, ymax in model_intervals], dtype=np.int32)
	names = np.array([name for name, time, xmin, xmax, ymin, ymax in model_intervals], dtype=str)
	num_objects = len(names)
	pred_bbox = {"bboxes":bboxes, "names":names, "times":times, "num_objects":num_objects}
	logging.debug('pred_bbox: %s', pred_bbox) ## DEBUG
	
	# crop each detection and save it as new image
	crop_path = os.path.join(output_path + '.crop')#, image_name)
	try:
		os.makedirs(crop_path)
	except FileExistsError:
		pass
	logging.info(' - Cropping images using (landmark) adjusted coords') ## INFO
	crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path)
	
	# draw colored boxes on image
	logging.info(' - Drawing crops for manual verification') ## INFO
	image = draw_bbox(original_image, pred_bbox)
	
	image = Image.fromarray(image.astype(np.uint8))
	image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
	cv2.imwrite(output_path + '.detection.png', image)
	
	logging.info('Done cropping tests from frame') ## INFO



# Extract mean value of RGB channels combined for a given image
# 
# Parameters:
#	image_filename	image file to get average RGB value for
def extract_colors(image_filename):
	pic = imageio.imread(image_filename)
	R = pic[ :, :, 0]
	G = pic[ :, :, 1]
	B = pic[ :, :, 2]
	meanR = np.mean(R)
	meanG = np.mean(G)
	meanB = np.mean(B)
	return ((meanR+meanG+meanB)/3)



# function for cropping each detection and saving as new image
def crop_objects(img, data, path, crop_offset=0):
	
	#data: bboxes, names, times, num_objects
	bboxes = data["bboxes"]
	names = data["names"]
	num_objects = data["num_objects"]
	
	for i in range(num_objects):
		# get box coords
		xmin, ymin, xmax, ymax = bboxes[i]
		
		# crop detection from image (take an additional x pixels around all edges; default 0)
		cropped_img = img[int(ymin)-crop_offset:int(ymax)+crop_offset, int(xmin)-crop_offset:int(xmax)+crop_offset]
		
		# construct image name and join it to path for saving crop properly
		img_name = names[i] + '.png'
		img_path = os.path.join(path, img_name )
		
		# save image
		cv2.imwrite(img_path, cropped_img)



def draw_bbox(image, data, show_label=True):
	image = np.copy(image)
	#data: bboxes, names, times, num_objects
	bboxes = data["bboxes"]
	names = data["names"]
	num_objects = data["num_objects"]
	
	image_h, image_w, _ = image.shape
	hsv_tuples = [(1.0 * x / num_objects, 1., 1.) for x in range(num_objects)]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
	
	random.seed(0)
	random.shuffle(colors)
	random.seed(None)
	
	for i in range(num_objects):
		coor = bboxes[i]
		fontScale = 0.5
		class_name = names[i]
		bbox_color = colors[i]
		bbox_thick = int(0.6 * (image_h + image_w) / 600)
		c1, c2 = (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3]))
		cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
		
		if show_label:
			bbox_mess = '%s' % (class_name)
			t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
			c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
			cv2.rectangle(image, c1, c3, bbox_color, -1) #filled
			cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
					fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
	return image



