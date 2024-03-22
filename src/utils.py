import logging
import sys

import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf

from src.Utilities.Video_Results import Video_Results
from src.Utilities.constants import csv_headers
from src.config import cfg
import csv
from src.Utilities.color_space_values import color_space_values


def load_freeze_layer(model='yolov4', tiny=False):
    if tiny:
        if model == 'yolov3':
            freeze_layouts = ['conv2d_9', 'conv2d_12']
        else:
            freeze_layouts = ['conv2d_17', 'conv2d_20']
    else:
        if model == 'yolov3':
            freeze_layouts = ['conv2d_58', 'conv2d_66', 'conv2d_74']
        else:
            freeze_layouts = ['conv2d_93', 'conv2d_101', 'conv2d_109']
    return freeze_layouts


def load_weights(model, weights_file, model_name='yolov4', is_tiny=False):
    if is_tiny:
        if model_name == 'yolov3':
            layer_size = 13
            output_pos = [9, 12]
        else:
            layer_size = 21
            output_pos = [17, 20]
    else:
        if model_name == 'yolov3':
            layer_size = 75
            output_pos = [58, 66, 74]
        else:
            layer_size = 110
            output_pos = [93, 101, 109]
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' % j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    # assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def load_config(tiny, model, class_file_name):
    if tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = get_anchors(cfg.YOLO.ANCHORS_TINY, tiny)
        XYSCALE = cfg.YOLO.XYSCALE_TINY if model == 'yolov4' else [1, 1]
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if model == 'yolov4':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS, tiny)
        elif model == 'yolov3':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS_V3, tiny)
        XYSCALE = cfg.YOLO.XYSCALE if model == 'yolov4' else [1, 1, 1]
    NUM_CLASS = len(read_class_names(class_file_name))

    return STRIDES, ANCHORS, NUM_CLASS, XYSCALE


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchors_path, tiny=False):
    anchors = np.array(anchors_path)
    if tiny:
        return anchors.reshape(2, 3, 2)
    else:
        return anchors.reshape(3, 3, 2)


def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


# helper function to convert bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
    return bboxes


def bbox_iou(bboxes1, bboxes2):
    """
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    return iou


def bbox_giou(bboxes1, bboxes2):
    """
    Generalized IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)

    return giou


def bbox_ciou(bboxes1, bboxes2):
    """
    Complete IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up

    c_2 = enclose_section[..., 0] ** 2 + enclose_section[..., 1] ** 2

    center_diagonal = bboxes2[..., :2] - bboxes1[..., :2]

    rho_2 = center_diagonal[..., 0] ** 2 + center_diagonal[..., 1] ** 2

    diou = iou - tf.math.divide_no_nan(rho_2, c_2)

    v = (
                (
                        tf.math.atan(
                            tf.math.divide_no_nan(bboxes1[..., 2], bboxes1[..., 3])
                        )
                        - tf.math.atan(
                    tf.math.divide_no_nan(bboxes2[..., 2], bboxes2[..., 3])
                )
                )
                * 2
                / np.pi
        ) ** 2

    alpha = tf.math.divide_no_nan(v, 1 - iou + v)

    ciou = diou - alpha * v

    return ciou


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bbox_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)


def unfreeze_all(model, frozen=False):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            unfreeze_all(l, frozen)


def write_rgb_vals_to_csv(csv_file_path, video_results):
    logging.info('Video Results')
    logging.info(video_results)
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)
        for video_name in video_results:
            video_result = video_results[video_name]
            row = [video_name]
            for shift in video_result:
                shift_result:video_result = video_result[shift]
                row_shift = [shift_result.bilirubin.red, shift_result.bilirubin.green, shift_result.bilirubin.blue,
                             shift_result.bilirubin.rgb_score, shift_result.bilirubin.l_star,
                             shift_result.bilirubin.a_star, shift_result.bilirubin.b_star, shift_result.bilirubin.cyan,
                             shift_result.bilirubin.yellow, shift_result.bilirubin.magenta,
                             shift_result.bilirubin.key_black,
                             shift_result.blood.red, shift_result.blood.green, shift_result.blood.blue,
                             shift_result.blood.rgb_score, shift_result.blood.l_star,
                             shift_result.blood.a_star, shift_result.blood.b_star, shift_result.blood.cyan,
                             shift_result.blood.yellow, shift_result.blood.magenta,
                             shift_result.blood.key_black,
                             shift_result.glucose.red, shift_result.glucose.green, shift_result.glucose.blue,
                             shift_result.glucose.rgb_score, shift_result.glucose.l_star,
                             shift_result.glucose.a_star, shift_result.glucose.b_star, shift_result.glucose.cyan,
                             shift_result.glucose.yellow, shift_result.glucose.magenta,
                             shift_result.glucose.key_black,
                             shift_result.ketone.red, shift_result.ketone.green, shift_result.ketone.blue,
                             shift_result.ketone.rgb_score, shift_result.ketone.l_star,
                             shift_result.ketone.a_star, shift_result.ketone.b_star, shift_result.ketone.cyan,
                             shift_result.ketone.yellow, shift_result.ketone.magenta,
                             shift_result.ketone.key_black,
                             shift_result.leukocytes.red, shift_result.leukocytes.green, shift_result.leukocytes.blue,
                             shift_result.leukocytes.rgb_score, shift_result.leukocytes.l_star,
                             shift_result.leukocytes.a_star, shift_result.leukocytes.b_star, shift_result.leukocytes.cyan,
                             shift_result.leukocytes.yellow, shift_result.leukocytes.magenta,
                             shift_result.leukocytes.key_black,
                             shift_result.nitrite.red, shift_result.nitrite.green, shift_result.nitrite.blue,
                             shift_result.nitrite.rgb_score, shift_result.nitrite.l_star,
                             shift_result.nitrite.a_star, shift_result.nitrite.b_star, shift_result.nitrite.cyan,
                             shift_result.nitrite.yellow, shift_result.nitrite.magenta,
                             shift_result.nitrite.key_black,
                             shift_result.ph.red, shift_result.ph.green, shift_result.ph.blue,
                             shift_result.ph.rgb_score, shift_result.ph.l_star,
                             shift_result.ph.a_star, shift_result.ph.b_star, shift_result.ph.cyan,
                             shift_result.ph.yellow, shift_result.ph.magenta,
                             shift_result.ph.key_black,
                             shift_result.protein.red, shift_result.protein.green, shift_result.protein.blue,
                             shift_result.protein.rgb_score, shift_result.protein.l_star,
                             shift_result.protein.a_star, shift_result.protein.b_star, shift_result.protein.cyan,
                             shift_result.protein.yellow, shift_result.protein.magenta,
                             shift_result.protein.key_black,
                             shift_result.specific_gravity.red, shift_result.specific_gravity.green,
                             shift_result.specific_gravity.blue,
                             shift_result.specific_gravity.rgb_score, shift_result.specific_gravity.l_star,
                             shift_result.specific_gravity.a_star, shift_result.specific_gravity.b_star,
                             shift_result.specific_gravity.cyan,
                             shift_result.specific_gravity.yellow, shift_result.specific_gravity.magenta,
                             shift_result.specific_gravity.key_black,
                             shift_result.urobilinogen.red, shift_result.urobilinogen.green,
                             shift_result.urobilinogen.blue,
                             shift_result.urobilinogen.rgb_score, shift_result.urobilinogen.l_star,
                             shift_result.urobilinogen.a_star, shift_result.urobilinogen.b_star,
                             shift_result.urobilinogen.cyan,
                             shift_result.urobilinogen.yellow, shift_result.urobilinogen.magenta,
                             shift_result.urobilinogen.key_black,]
                row.extend(row_shift)
            writer.writerow(row)


def update_standard_deviation(standard_values: color_space_values, color_value: color_space_values,
                              deviation_from_standards: color_space_values):
    deviation_from_standards.red += standard_values.red - color_value.red
    deviation_from_standards.blue += standard_values.blue - color_value.blue
    deviation_from_standards.green += standard_values.green - color_value.green
    deviation_from_standards.rgb_score += standard_values.rgb_score - color_value.rgb_score
    deviation_from_standards.l_star += standard_values.l_star - color_value.l_star
    deviation_from_standards.a_star += standard_values.a_star - color_value.a_star
    deviation_from_standards.b_star += standard_values.b_star - color_value.b_star
    deviation_from_standards.cyan += standard_values.cyan - color_value.cyan
    deviation_from_standards.magenta += standard_values.magenta - color_value.magenta
    deviation_from_standards.yellow += standard_values.yellow - color_value.yellow
    deviation_from_standards.key_black += standard_values.key_black - color_value.key_black


def shift_hue(image, shift):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = image_hsv[:, :, 0]
    s = image_hsv[:, :, 1]
    v = image_hsv[:, :, 2]
    h_shifted = np.mod(h + shift, 360).astype(np.uint8)
    hue_shifted_image_hsv = cv2.merge([h_shifted, s, v])
    hue_shifted_image_bgr = cv2.cvtColor(hue_shifted_image_hsv, cv2.COLOR_HSV2BGR)
    return hue_shifted_image_bgr


def adjust_color_space_values(test_color_space_values: color_space_values,
                              deviation_from_standards: color_space_values):
    test_color_space_values.red += deviation_from_standards.red
    test_color_space_values.green += deviation_from_standards.green
    test_color_space_values.blue += deviation_from_standards.blue
    test_color_space_values.rgb_score += deviation_from_standards.rgb_score
    test_color_space_values.l_star += deviation_from_standards.l_star
    test_color_space_values.a_star += deviation_from_standards.a_star
    test_color_space_values.b_star += deviation_from_standards.b_star
    test_color_space_values.cyan += deviation_from_standards.cyan
    test_color_space_values.yellow += deviation_from_standards.yellow
    test_color_space_values.magenta += deviation_from_standards.magenta
    test_color_space_values.key_black += deviation_from_standards.key_black
    return test_color_space_values

def get_filename_from_path(path:str):
    names = path.split('\\')
    return names[-1]
