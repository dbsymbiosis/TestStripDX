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



# Crop tests from provided image file
#
# Parameters:
#	image_path	path to input image
#	output		path to output folder
#	intervals	intervals to extract
#	input_size	resize images to (default: 416)
def crop_test_strip(image_path, output, intervals,
		input_size=416):
	
	original_image = cv2.imread(image_path)
	original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
	
	image_data = cv2.resize(original_image, (input_size, input_size))
	image_data = image_data / 255.
	
	# hold all detection data in one variable
	bboxes = np.array([ [xmin, ymin, xmax, ymax] for name, time, xmin, xmax, ymin, ymax in intervals], dtype=np.float32)
	times = np.array([ time for name, time, xmin, xmax, ymin, ymax in intervals], dtype=np.int32)
	names = np.array([ name for name, time, xmin, xmax, ymin, ymax in intervals], dtype=str)
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
	image = draw_bbox(original_image, pred_bbox)
	
	image = Image.fromarray(image.astype(np.uint8))
	image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
	cv2.imwrite(output + '.detection.png', image)



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
	
	boxes, names, times, num_objects = data
	
	for i in range(num_objects):
		# get box coords
		xmin, ymin, xmax, ymax = boxes[i]
		
		# crop detection from image (take an additional x pixels around all edges; default 0)
		cropped_img = img[int(ymin)-crop_offset:int(ymax)+crop_offset, int(xmin)-crop_offset:int(xmax)+crop_offset]
		
		# construct image name and join it to path for saving crop properly
		img_name = names[i] + '.png'
		img_path = os.path.join(path, img_name )
		
		# save image
		cv2.imwrite(img_path, cropped_img)



def draw_bbox(image, bboxes, info = False, show_label=True):
	out_boxes, names, times, num_objects = bboxes
	
	num_classes = num_objects
	image_h, image_w, _ = image.shape
	hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
	
	random.seed(0)
	random.shuffle(colors)
	random.seed(None)
	
	for i in range(num_objects):
		coor = out_boxes[i]
		fontScale = 0.5
		class_name = names[i]
		bbox_color = colors[i]
		bbox_thick = int(0.6 * (image_h + image_w) / 600)
		c1, c2 = (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3]))
		cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
		
		if info:
			print("Object found: {}, BBox Coords (xmin, ymin, xmax, ymax): {}, {}, {}, {} ".format(class_name, coor[0], coor[1], coor[2], coor[3]))
		
		if show_label:
			bbox_mess = '%s' % (class_name)
			t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
			c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
			cv2.rectangle(image, c1, c3, bbox_color, -1) #filled
			cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
					fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
	return image



