#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : config.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:06:54
#   Description :
#
#================================================================

from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

# Set the class name
# __C.YOLO.CLASSES              = "./data/classes/coco.names"
__C.YOLO.CLASSES              = '/home/luigy/luigy/petrobras/drive/deteccao_de_objetos/labelimg/poco_2_and_ee/forYolo/mydataset.names'

__C.YOLO.ANCHORS              = "./data/anchors/basline_anchors.txt"
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5

# Train options
__C.TRAIN                     = edict()



# __C.TRAIN.ANNOT_PATH          = "./data/dataset/yymnist_train.txt"
__C.TRAIN.ANNOT_PATH          = '/home/luigy/luigy/petrobras/drive/deteccao_de_objetos/labelimg/poco_2_and_ee/forYolo/mydataset_train.txt'
# __C.TRAIN.ANNOT_PATH          = '/home/luigy/luigy/petrobras/drive/deteccao_de_objetos/labelimg/poco_ee/forYolo/mydataset_train_with_aug.txt'

# __C.TRAIN.BATCH_SIZE          = 4
__C.TRAIN.BATCH_SIZE          = 4
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = [416]
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 20
__C.TRAIN.EPOCHS              = 100



# TEST options
__C.TEST                      = edict()

# __C.TEST.ANNOT_PATH           = "./data/dataset/yymnist_test.txt"
# __C.TEST.ANNOT_PATH           = '/home/luigy/luigy/petrobras/luigy/tf2-yolov3/4-Object_Detection/YOLOV3/data/dataset/mydataset_test.txt'
# __C.TEST.ANNOT_PATH           = '/home/luigy/luigy/petrobras/drive/deteccao_de_objetos/labelimg/poco_ee/forYolo/mydataset_test_with_aug.txt'
__C.TEST.ANNOT_PATH           = '/home/luigy/luigy/petrobras/drive/deteccao_de_objetos/labelimg/poco_2_and_ee/forYolo/mydataset_test.txt'
__C.TEST.BATCH_SIZE           = 4
# __C.TEST.BATCH_SIZE           = 16
# __C.TEST.INPUT_SIZE           = 532
__C.TEST.INPUT_SIZE           = 416
__C.TEST.DATA_AUG             = False
# __C.TEST.DECTECTED_IMAGE_PATH = "/home/luigy/luigy/petrobras/drive/deteccao_de_objetos/labelimg/poco_ee/forYolo/results/detection/"
__C.TEST.DECTECTED_IMAGE_PATH = "/home/luigy/luigy/petrobras/luigy/tf2-yolov3/4-Object_Detection/YOLOV3/results" #error
__C.TEST.SCORE_THRESHOLD      = 0.8
__C.TEST.IOU_THRESHOLD        = 0.8


