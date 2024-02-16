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
from roboflow import Roboflow
from ultralytics import YOLO

from src.detector import detect_test_strip


# Crop tests from provided image file
#
# Parameters:
#	image_path	 	path to input image
#	output		 	path to output folder
#       model_detector_path	path to Tensorflow detector file (e.g., 'models/teststrips.detector')
#	crop			crop detected regions out of image
#	conf			object confidence threshold for detection
#	imgsz			image size as scalar or (h, w) list, i.e. (640, 480)
def run_detector_on_image(image_path, output_path,
                          model_detector_path,
                          crop=True,
                          conf=1.00,
                          imgsz=640):
    logging.info('Start cropping tests from frame: %s', image_path)  ## INFO

    ## Import model
    model = YOLO(model_detector_path)

    ## Get results from detector
    model.predict(image_path, {'conf': conf, 'imgsz': imgsz})

    original_image = cv2.imread(image_path)

    # Search for landmark using ML
    logging.info(' - Using ML to search for landmark objects')  ## INFO
    logging.debug('In frame: %s', image_path)  ## DEBUG
    logging.debug('Out prefix: %s', output_path)  ## DEBUG
    # Check if we found the landmark in the image and extract it coordinates
    landmark_found, l_xmin, l_ymin, l_xmax, l_ymax = check_landmark(original_image,
                                                                    model_detector_path, model_names_path, model_names,
                                                                    model_landmark_bounds,
                                                                    output_path)
    if not landmark_found:
        logging.warning(
            'Landmark ML (%s: xmin:%s, xmax:%s, ymin:%s, ymax:%s) was outside the expected bounds (xmin:%s, xmax:%s, ymin:%s, ymax:%s). This might mean that the video has an unexpected rotation or that the strip might not be correctly positioned in the holder.',
            model_landmark_bounds["name"], l_xmin, l_xmax, l_ymin, l_ymax,
            model_landmark_bounds["xmin"], model_landmark_bounds["xmax"], model_landmark_bounds["ymin"],
            model_landmark_bounds["ymax"])  # WARNING

    # hold all detection data in one variable
    bboxes = np.array([[
        l_xmin + xmin,
        l_ymin + ymin,
        l_xmin + xmax,
        l_ymin + ymax
    ] for name, time, xmin, xmax, ymin, ymax in model_intervals], dtype=np.float32)
    times = np.array([time for name, time, xmin, xmax, ymin, ymax in model_intervals], dtype=np.int32)
    names = np.array([name for name, time, xmin, xmax, ymin, ymax in model_intervals], dtype=str)
    num_objects = len(names)
    pred_bbox = {"bboxes": bboxes, "names": names, "times": times, "num_objects": num_objects}
    logging.debug('pred_bbox: %s', pred_bbox)  ## DEBUG

    # draw colored boxes on image
    logging.info(' - Drawing crops for manual verification')  ## INFO
    image = draw_bbox(original_image, pred_bbox)

    image = Image.fromarray(image.astype(np.uint8))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path + '.detection.png', image)

    # crop each detection and save it as new image
    crop_path = os.path.join(output_path + '.crop')  # , image_name)
    try:
        os.makedirs(crop_path)
    except FileExistsError:
        pass
    logging.info(' - Cropping images using (landmark) adjusted coords')  ## INFO
    crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path)

    logging.info('Done cropping tests from frame')  ## INFO


# Check for landmark in frame. Return True if it is in the correct position (i.e., frame is correctly orientated).
#
# Parameters:
#	frame			frame to check
#	model_detector_path	path to Tensorflow detector file (e.g., 'models/teststrips.detector')
#	model_names_path	path to names file (e.g., 'models/teststrips.names')
#	model_names		names of model objects
#	model_landmark_bounds	dict of landmark features to check for in image
#	output_path		render ML pred to image. Useful for debugging (if None, dont render)
#
# Returns:
#	found_landmark 				either True (Landmark was found in correct position) or False (Landmark wasnt found in correct position - or wasnt found at all)
#	l_xmin, l_ymin, l_xmax, l_ymax		bounds of landmark from ML prediction
def check_landmark(frame, model_detector_path, model_names_path, model_names, model_landmark_bounds, output_path=None):
    # Run ML to predict landmark
    pred_bbox = detect_test_strip(model_detector_path, model_names_path, model_names, frame)
    bboxes = pred_bbox["bboxes"]
    names = pred_bbox["names"]
    num_objects = pred_bbox["num_objects"]
    l_xmin, l_ymin, l_xmax, l_ymax = 460, 100, 530, 140

    # draw colored boxes on image for ML detections (used for debugging)
    if output_path is not None:
        ML_frame = draw_bbox(frame, pred_bbox)
        ML_frame = Image.fromarray(ML_frame.astype(np.uint8))
        ML_frame = cv2.cvtColor(np.array(ML_frame), cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path + '.ML_detection.png', ML_frame)

    # Check each of the predicted features and check if they are the one we want.
    for i in range(num_objects):
        if names[i] == model_landmark_bounds["name"]:
            logging.debug('Landmark found')  ## DEBUG
            l_xmin, l_ymin, l_xmax, l_ymax = bboxes[i]
            break
    else:
        logging.debug(
            'Failed to find landmark in frame. Falling back to default coords, these are a rought approximation but will likely be wrong. Please double check these results.')  ## DEBUG
        return (False, l_xmin, l_ymin, l_xmax, l_ymax)

    # Check if landmark is where we expect - return False if its out of place
    if l_xmin < model_landmark_bounds["xmin"] or \
            l_xmin > model_landmark_bounds["xmax"] or \
            l_ymin < model_landmark_bounds["ymin"] or \
            l_ymin > model_landmark_bounds["ymax"]:
        logging.debug(
            'Landmark ML (%s: xmin:%s, xmax:%s, ymin:%s, ymax:%s) was OUT side the expected bounds (xmin:%s, xmax:%s, ymin:%s, ymax:%s).',
            model_landmark_bounds["name"], l_xmin, l_xmax, l_ymin, l_ymax,
            model_landmark_bounds["xmin"], model_landmark_bounds["xmax"], model_landmark_bounds["ymin"],
            model_landmark_bounds["ymax"])  # DEBUG
        return (False, l_xmin, l_ymin, l_xmax, l_ymax)
    else:
        logging.debug(
            'Landmark ML (%s: xmin:%s, xmax:%s, ymin:%s, ymax:%s) was IN side the expected bounds (xmin:%s, xmax:%s, ymin:%s, ymax:%s).',
            model_landmark_bounds["name"], l_xmin, l_xmax, l_ymin, l_ymax,
            model_landmark_bounds["xmin"], model_landmark_bounds["xmax"], model_landmark_bounds["ymin"],
            model_landmark_bounds["ymax"])  # DEBUG
        return (True, l_xmin, l_ymin, l_xmax, l_ymax)


# Extract mean value of RGB channels combined for a given image
# 
# Parameters:
#	image_filename	image file to get average RGB value for
def extract_colors(image_filename):
    pic = imageio.imread(image_filename)
    R = pic[:, :, 0]
    G = pic[:, :, 1]
    B = pic[:, :, 2]
    meanR = np.mean(R)
    meanG = np.mean(G)
    meanB = np.mean(B)
    score = (meanR + meanG + meanB) / 3
    return ({'score': score, 'R': meanR, 'G': meanG, 'B': meanB})


# function for cropping each detection and saving as new image
def crop_objects(img, data, path, crop_offset=0):
    # data: bboxes, names, times, num_objects
    bboxes = data["bboxes"]
    names = data["names"]
    num_objects = data["num_objects"]

    for i in range(num_objects):
        # get box coords
        xmin, ymin, xmax, ymax = bboxes[i]
        logging.debug('Cropping ojbect no:%s at coords xmin:%s, ymin:%s, xmax:%s, ymax:%s', i, xmin, ymin, xmax,
                      ymax)  ## DEBUG

        # crop detection from image (take an additional x pixels around all edges; default 0)
        cropped_img = img[int(ymin) - crop_offset:int(ymax) + crop_offset,
                      int(xmin) - crop_offset:int(xmax) + crop_offset]

        # construct image name and join it to path for saving crop properly
        img_name = names[i] + '.png'
        img_path = os.path.join(path, img_name)

        # save image
        cv2.imwrite(img_path, cropped_img)


def draw_bbox(image, data, show_label=True):
    image = np.copy(image)
    # data: bboxes, names, times, num_objects
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
        logging.debug('name:%s; color:%s; coords:%s', class_name, bbox_color, coor)  ## DEBUG
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3]))
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s' % (class_name)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, c3, bbox_color, -1)  # filled
            cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image

def predict_image_save_boxes(image_path: str, model_path: str, output_text_path: str, roboflow_api_key: str,
                             project_name: str):
    logging.info(model_path)
    model = YOLO(model_path)
    logging.info('Hi')
    predictions = model.predict(source=image_path, stream=False)
    logging.info(f'output dir: {output_text_path}')
    open(output_text_path, 'w').close()
    with open(output_text_path, '+w') as file:
        logging.error(f'Opened file: {output_text_path}')
        for idx, prediction in enumerate(predictions[0].boxes.xywhn):  # change final attribute to desired box format
            cls = int(predictions[0].boxes.cls[idx].item())
            # path = predictions[0].path
            class_name = model.names[cls]
            logging.error(class_name)
            file.write(
                f"{cls} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()}\n")
    logging.error('Hi')
    rf = Roboflow(api_key=roboflow_api_key)
    project = rf.workspace().project(project_name)
    logging.info(
        project.upload(image_path=image_path, annotation_path=output_text_path, split='train'))