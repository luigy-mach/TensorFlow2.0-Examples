#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-07-19 10:29:34
#   Description :
#
#================================================================

import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOv3, decode


INPUT_SIZE   = 416
NUM_CLASS    = len(utils.read_class_names(cfg.YOLO.CLASSES))
CLASSES      = utils.read_class_names(cfg.YOLO.CLASSES)


DIR_RESULTS = cfg.TRAIN.DIR_train

predicted_dir_path = 'results/predicted'
ground_truth_dir_path = 'results/ground-truth'
imgs_dir_path = 'results/imgs'
imgs_clean_dir_path = 'results/imgs_clean'


predicted_dir_path = os.path.join(DIR_RESULTS,predicted_dir_path)
ground_truth_dir_path = os.path.join(DIR_RESULTS,ground_truth_dir_path)
imgs_dir_path = os.path.join(DIR_RESULTS,imgs_dir_path)
imgs_clean_dir_path = os.path.join(DIR_RESULTS,imgs_clean_dir_path)


if not os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): print("error dont exist %"%cfg.TEST.DECTECTED_IMAGE_PATH)

if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
if os.path.exists(imgs_dir_path): shutil.rmtree(imgs_dir_path)
if os.path.exists(imgs_clean_dir_path): shutil.rmtree(imgs_clean_dir_path)
# if not os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): print("error  dont exist cfg.TEST.DECTECTED_IMAGE_PATH"); exit(1);

os.makedirs(imgs_dir_path)
os.makedirs(predicted_dir_path)
os.makedirs(ground_truth_dir_path)
os.makedirs(imgs_clean_dir_path)

# Build Model
input_layer  = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
feature_maps = YOLOv3(input_layer)

bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(input_layer, bbox_tensors)
# model.load_weights("./yolov3")
model.load_weights(os.path.join(DIR_RESULTS,'yolov3-480epoch'))
# model.load_weights("/home/luigy/luigy/petrobras/luigy/tf2-yolov3/4-Object_Detection/YOLOV3/train_weights/yolov3_99")

with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:
    for num, line in enumerate(annotation_file):
        annotation = line.strip().split()
        image_path = annotation[0]
        image_name = image_path.split('/')[-1]
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

        if len(bbox_data_gt) == 0:
            bboxes_gt=[]
            classes_gt=[]
        else:
            bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
        # ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')
        # ground_truth_path = os.path.join(ground_truth_dir_path, str(num)+"_"+os.path.splitext(image_name)[0]+'.txt')
        ground_truth_path = os.path.join(ground_truth_dir_path, os.path.splitext(image_name)[0]+'.txt')

        print('=> ground truth of %s:' % image_name)
        num_bbox_gt = len(bboxes_gt)
        with open(ground_truth_path, 'w') as f:
            for i in range(num_bbox_gt):
                class_name = CLASSES[classes_gt[i]]
                xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())
        print('=> predict result of %s:' % image_name)
        # predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
        # predict_result_path = os.path.join(predicted_dir_path, str(num)+"_"+os.path.splitext(image_name)[0]+'.txt')
        predict_result_path = os.path.join(predicted_dir_path, os.path.splitext(image_name)[0]+'.txt')
        
        # Predict Process
        image_size = image.shape[:2]
        image_data = utils.image_preporcess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        # print("***************************************")
        # print(type(pred_bbox))
        # print("***************************************")
        # pred_bbox2 = np.array(pred_bbox)
        # print(type(pred_bbox2))
        # print(pred_bbox2.shape)
        # print(pred_bbox2)
        # print("***************************************")
        # print(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
        pred_xywh = pred_bbox[:,0:4]
        print(pred_xywh)
        pred_conf = pred_bbox[:,4]
        print(pred_conf)
        pred_prob = pred_bbox[:,5:]
        print(pred_prob)
        print("***************************************")
        bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
        bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')


        if imgs_dir_path is not None:
            # temp = list(map(int, bboxes_gt[i]))
            # temp.append
            # image = utils.draw_bbox_gt(image, list(map(int, bboxes_gt[i])), classes_gt)
            cv2.imwrite(os.path.join(imgs_clean_dir_path,image_name), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            image = utils.draw_bbox(image, bboxes)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(imgs_dir_path,image_name), image)

        with open(predict_result_path, 'w') as f:
            for bbox in bboxes:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = CLASSES[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = list(map(str, coor))
                bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())

