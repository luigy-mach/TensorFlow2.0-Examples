#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-07-18 09:18:54
#   Description :
#
#================================================================

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import YOLOv3, decode, compute_loss
from core.config import cfg

import json

path_save_weights = cfg.TRAIN.DIR_train
# path_save_weights= 'train_weights_1class_with_DAug_v8/'





if not os.path.isdir(path_save_weights):
    os.makedirs(path_save_weights)
trainset = Dataset('train')
logdir = os.path.join(path_save_weights,'data/log')

if not os.path.isdir(logdir):
    os.makedirs(logdir)

steps_per_epoch = len(trainset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch

input_tensor = tf.keras.layers.Input([416, 416, 3])
conv_tensors = YOLOv3(input_tensor)

output_tensors = []
for i, conv_tensor in enumerate(conv_tensors):
    pred_tensor = decode(conv_tensor, i)
    output_tensors.append(conv_tensor)
    output_tensors.append(pred_tensor)

model = tf.keras.Model(input_tensor, output_tensors)
optimizer = tf.keras.optimizers.Adam()
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)


def train_step(image_data, target,file):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss=conf_loss=prob_loss=0

        # optimizing process
        for i in range(3):
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f " %(global_steps, optimizer.lr.numpy(),
                                                          giou_loss, conf_loss,
                                                          prob_loss, total_loss))
        file.write("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f \n" %(global_steps, optimizer.lr.numpy(),
                                                          giou_loss, conf_loss,
                                                          prob_loss, total_loss))       # update learning rate
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps *cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()




file_parameters = os.path.join(path_save_weights,"file_parameters_train.txt")
file_console_outputs = os.path.join(path_save_weights,"file_console_outputs.txt")
with open(file_parameters, "w") as file:
    file.write(json.dumps(cfg, indent=4))



with open(file_console_outputs,"w") as file2:
    for epoch in range(cfg.TRAIN.EPOCHS):
        for image_data, target in trainset:
            train_step(image_data, target,file2)
        # model.save_weights("train_weights_1class_with_DAug_v5/yolov3-{:03d}".format(epoch))
        print("/////////////////////////////////////////////////////////////////////")
        print("///////////////////////save epoch {:03d}".format(epoch+1))
        print("/////////////////////////////////////////////////////////////////////")
        
        file2.write("/////////////////////////////////////////////////////////////////////\n")
        file2.write("///////////////////////save epoch {:03d} \n".format(epoch+1))
        file2.write("/////////////////////////////////////////////////////////////////////\n")
        if (epoch+1)%cfg.TRAIN.SAVE_WEIGTHS_EVERY_EPOCHS == 0:
            model.save_weights(os.path.join(path_save_weights,"yolov3-{:03d}epoch".format(epoch+1)))

