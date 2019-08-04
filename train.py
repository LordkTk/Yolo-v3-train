# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:23:13 2019

@author: cfd_Liu
"""

import tensorflow as tf
import numpy as np
import os
import random
import time
from train_utils import post_process, decode, proc_anchors, cal_loss, cal_y, build_net, load_img, div_train_test
from data_aug import load_aug_img
from mAP import get_mAP
tf.reset_default_graph()

class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 
               'sofa', 'train', 'tvmonitor']

def train_yolo(resume = False, predict = False):
    imgInfo_train = np.load('./data/dataset_train.npy').item()
    imgInfo_test = np.load('./data/dataset_test.npy').item()
    trainList = np.load('./data/trainList.npy').item()[1]
    testList = np.load('./data/testList.npy').item()[1]
    testLossList = testList[:5]
    path_train = np.load('./data/path_train.npy').item()[1]
    path_test = np.load('./data/path_test.npy').item()[1]
    
    multiscale = False
    imgSize = 416  

    anchors = np.array([33.3821, 44.4139, 59.3492, 115.195, 159.471, 117.935, 87.7129, 208.548, 209.227, 227.445, 126.862, 320.245, 352.677, 202.811, 243.887, 357.922, 374.698, 370.688]).reshape([9, 2]).astype(np.float32)
    anchorCoor_test = proc_anchors(anchors, imgSize)
    
    x = tf.placeholder(tf.float32, [None, None, None, 3])
    y1 = tf.placeholder(tf.float32, [None, None, None, 3, 25])
    y2 = tf.placeholder(tf.float32, [None, None, None, 3, 25])
    y3 = tf.placeholder(tf.float32, [None, None, None, 3, 25])
    bbox = tf.placeholder(tf.float32, [None, None, 4])
    anchor1 = tf.placeholder(tf.float32, [None, None, 3, 4])
    anchor2 = tf.placeholder(tf.float32, [None, None, 3, 4])
    anchor3 = tf.placeholder(tf.float32, [None, None, 3, 4])
    
    is_training = tf.placeholder(tf.bool)
    
    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    lr_init = 1e-4
    
    sess = tf.Session()
    exp_name = 'exp3'
    if predict:
        out = build_net(x, is_training)
        v_list = tf.contrib.framework.get_variables_to_restore()
        saver = tf.train.Saver(v_list)
        file = tf.train.latest_checkpoint('./Weights/%s/fst_stage/' %exp_name)
        saver.restore(sess, file)
        
        pathList = []
        for test in testList[:10]:
            pathList.append(os.path.join(path_test, test))
#        pathList = ['C:/Users/cfd_Liu/Desktop/Machine Learning/Code/PracticeCode/TensorFlow Learning/OpenCV/LaneDet/YOLO/img/sample_dog.jpg']
        post_process(sess, x, is_training, out, pathList, imgSize, 20, class_names, anchors)
    else:
        out = build_net(x, is_training)
        
        epochWarm = 5
        batchSize = 4
        decayWarm_step = (len(trainList)//batchSize) * epochWarm if len(trainList)%batchSize==0 else (len(trainList)//batchSize + 1) * epochWarm
        
        epochFst = 95
        decayFst_step = (len(trainList)//batchSize) * epochFst if len(trainList)%batchSize==0 else (len(trainList)//batchSize + 1) * epochFst
        decay_rate = 0.01**(1/epochFst)
        save_name = 'fst_stage'
        epochMax = epochFst + epochWarm
        
        learn_rate = tf.cond(tf.less(global_step, decayWarm_step), 
                            lambda: tf.train.polynomial_decay(1e-10, global_step, decayWarm_step, lr_init),
                            lambda: tf.train.exponential_decay(lr_init, global_step - decayWarm_step, decayFst_step, decay_rate, staircase=True))
        
        loss = cal_loss(out, [y1, y2, y3], [anchor1, anchor2, anchor3], bbox, focal_loss=True, regularization = False)
        
        restore_vars = tf.contrib.framework.get_variables_to_restore(exclude=['head/det3_7', 'head/det2_7', 'head/det1_6', 'global_step'])
        update_vars = tf.contrib.framework.get_variables_to_restore(include=['head'])
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(learn_rate, 0.9)
#            optimizer = tf.train.AdamOptimizer(learn_rate)
            train_step = optimizer.minimize(loss[0], var_list = update_vars, global_step = global_step)
        if resume:
            saver = tf.train.Saver()
            file = tf.train.latest_checkpoint('./Weights/%s/fst_stage/' %exp_name)
            saver.restore(sess, file)
        else:
            sess.run(tf.global_variables_initializer())
            
            saver = tf.train.Saver(restore_vars)
            saver.restore(sess, './Weights/weightsInit.ckpt')
        
        'Put all tf operations out of loop! Or the tf computing graphs will add consistently'
        decode_op = decode(out, imgSize, 20, anchors)
        'Confirm the tf graphs have been built before loop with this function. But it would conflict with Saver()'
#        sess.graph.finalize()
        
        testLoss = []
        mAP0 = 0
        for epoch in range(epochMax):
            timeS = time.time()
            sampleList = trainList.copy()
            if multiscale and epoch >= epochWarm:
                imgSize_train = 32 * random.randint(10, 20)
            else:
                imgSize_train = 416
            anchorCoor = proc_anchors(anchors, imgSize_train)
            step = 0
            save = False
            
            while sampleList != []:
                batchList = []
                if len(sampleList) > batchSize:
                    randint = random.sample(range(0, len(sampleList)), batchSize)
                else:
                    randint = list(range(len(sampleList)))
                for i in randint:
                    batchList.append(sampleList[i])
                for i in range(len(randint)):
                    sampleList.remove(batchList[i])
                
                imgbatch, imgInfo = load_aug_img(path_train, batchList, imgSize_train, imgInfo_train)
                
                y, bbox_gr = cal_y(batchList, imgInfo, anchorCoor, imgSize_train)
                
                feed_dict = {x: imgbatch, y1: y[0], y2: y[1], 
                             y3: y[2], bbox: bbox_gr, anchor1: anchorCoor[0], 
                             anchor2: anchorCoor[1], anchor3: anchorCoor[2],
                             is_training: True}
                sess.run(train_step, feed_dict = feed_dict)
                
                if step % 100 == 0:
                    feed_dict[is_training] = False
                    print(sess.run(loss, feed_dict = feed_dict))
                step+= 1
            
            imgbatch = load_img(path_test, testLossList, imgSize)
            y, bbox_gr = cal_y(testLossList, imgInfo_test, anchorCoor_test, imgSize)
            feed_dict = {x: imgbatch, y1: y[0], y2: y[1], 
                         y3: y[2], bbox: bbox_gr, anchor1: anchorCoor_test[0], 
                         anchor2: anchorCoor_test[1], anchor3: anchorCoor_test[2],
                         is_training: False}
            testLoss.append(sess.run(loss, feed_dict = feed_dict))
            print('Test Loss:', testLoss[-1])
            timeE = time.time()
            print(epoch, timeE - timeS)
            if epoch >= epochWarm-1 and epoch%2 == 0:
                AP, mAP = get_mAP(sess, x, decode_op, is_training, out, class_names, imgInfo_test, testList, path_test, imgSize, anchors, batchSize)
                if mAP > mAP0:
                    save = True
                    mAP0 = mAP
                    print('------------------save epoch: %d, mAP: %f-----------------------' %(epoch, mAP))
            if save:
                tf.train.Saver().save(sess, './Weights/%s/%s/%s' % (exp_name, save_name, save_name), epoch)
        np.save('./Weights/%s/%s/%s' % (exp_name, save_name, save_name), np.array(testLoss))
        sess.close()
        
if __name__ == '__main__' :
#    div_train_test()
        
#    train_yolo(resume=True)
    train_yolo(predict=True)


