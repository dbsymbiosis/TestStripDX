import os
import cv2
import random
import numpy as np
import tensorflow as tf
import pytesseract
from src.config import cfg

# function to count objects, can return total classes or count per class
def count_objects(data, allowed_classes, class_names, by_class = False):
    boxes, scores, classes, num_objects = data

    #create dictionary to hold count of objects
    counts = dict()

    # if by_class = True then count objects per class
    if by_class:
        # loop through total number of objects found
        for i in range(num_objects):
            # grab class index and convert into corresponding class name
            class_index = int(classes[i])
            class_name = class_names[class_index]
            if class_name in allowed_classes:
                counts[class_name] = counts.get(class_name, 0) + 1
            else:
                continue

    # else count total objects found
    else:
        counts['total object'] = num_objects
    
    return counts

# function for cropping each detection and saving as new image
def crop_objects(img, data, path, crop_offset=0):
    boxes, names, times, num_objects = data
    #create dictionary to hold count of objects for image name
    counts = dict()
    for i in range(num_objects):
        # get box coords
        xmin, ymin, xmax, ymax = boxes[i]
        # crop detection from image (take an additional x pixels around all edges; default 0)
        cropped_img = img[int(ymin)-crop_offset:int(ymax)+crop_offset, int(xmin)-crop_offset:int(xmax)+crop_offset]
        # construct image name and join it to path for saving crop properly
        img_name = name[i] + '_' + str(counts[class_name]) + '.png'
        img_path = os.path.join(path, img_name )
        # save image
        cv2.imwrite(img_path, cropped_img)
        
# function to run general Tesseract OCR on any detections 
def ocr(img, data, class_names):
    boxes, scores, classes, num_objects = data
    for i in range(num_objects):
        # get class name for detection
        class_index = int(classes[i])
        class_name = class_names[class_index]
        # separate coordinates from box
        xmin, ymin, xmax, ymax = boxes[i]
        # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
        box = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
        # grayscale region within bounding box
        gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
        # threshold the image using Otsus method to preprocess for tesseract
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # perform a median blur to smooth image slightly
        blur = cv2.medianBlur(thresh, 3)
        # resize image to double the original size as tesseract does better with certain text size
        blur = cv2.resize(blur, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
        # run tesseract and convert image text to string
        try:
            text = pytesseract.image_to_string(blur, config='--psm 11 --oem 3')
            print("Class: {}, Text Extracted: {}".format(class_name, text))
        except: 
            text = None