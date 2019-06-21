# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:57:14 2019

@author: cfd_Liu
"""

import numpy as np
import tensorflow as tf
import cv2
import os
from train_utils import cal_iou, decode, build_net
import time
def load_img(pathList, imgSize):
    imgTotal = []
    for path in pathList:
        img = np.float32(cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2RGB)/255)
        img = cv2.resize(img, (imgSize, imgSize))[np.newaxis, :,:,:]
        imgTotal.append(img)
    imgTotal = np.concatenate(imgTotal, axis=0)
    return imgTotal
def post_process(class_probsTotal, obj_probsTotal, bboxTotal, batchSize):
    def arg_max(x):
        xmax = 0
        id1 = 0
        id2 = 0
        w1, w2 = x.shape
        for i in range(w1):
            for j in range(w2):
                if x[i, j] > xmax:
                    xmax = x[i, j]
                    id1 = i
                    id2 = j
        return id1, id2
    results = []
    for i in range(batchSize):
        #[:, 3]; [:, 3, 4]
        class_name = np.argmax(class_probsTotal[i], axis = 2)
        class_probs = np.max(class_probsTotal[i], axis = 2)
        obj_probs = obj_probsTotal[i]
        bbox = bboxTotal[i]
        bbox[bbox > 1] = 1
        bbox[bbox < 0] = 0
        confidence = class_probs * obj_probs
        conf = confidence.copy()
        confidence[confidence < 0.5] = 0 ######################################################################################
        
        det_indTotal = []
        while (np.max(confidence) != 0):
            id1, id2 = arg_max(confidence)
            bbox1_area = (bbox[id1, id2, 3]-bbox[id1, id2, 1])*(bbox[id1, id2, 2]-bbox[id1, id2, 0])
            sign = 0
            for coor in det_indTotal:
                if coor == None:
                    det_indTotal.append([id1, id2])
                else:
                    xi1 = max(bbox[id1, id2, 0], bbox[coor[0], coor[1], 0])
                    yi1 = max(bbox[id1, id2, 1], bbox[coor[0], coor[1], 1])
                    xi2 = min(bbox[id1, id2, 2], bbox[coor[0], coor[1], 2])
                    yi2 = min(bbox[id1, id2, 3], bbox[coor[0], coor[1], 3])
                    int_area = (max(yi2, 0) - max(yi1, 0)) * (max(xi2, 0) - max(xi1, 0))
                    bbox2_area = (bbox[coor[0], coor[1], 3] - bbox[coor[0], coor[1], 1]) * (bbox[coor[0], coor[1], 2] - bbox[coor[0], coor[1], 0])
                    uni_area = bbox1_area + bbox2_area - int_area
                    iou = int_area / uni_area
                    if iou > 0.5: ###########################################################################################
                        sign = 1
                        break
            if sign == 0:
                det_indTotal.append([id1, id2]) 
            confidence[id1, id2] = 0
        result = []
        for [id1, id2] in det_indTotal:
            result.append([class_name[id1,id2], conf[id1,id2], bbox[id1,id2,0], bbox[id1,id2,1], bbox[id1,id2,2], bbox[id1,id2,3]])
        results.append(result)
    return results

def get_mAP(sess, x, decode_op, is_training, out, class_names, imgInfo, testList, path, imgSize, anchors, batchSize):
    t1 = time.time()
    count = [[] for i in range(len(class_names))]
    obj_gr_count = np.zeros([len(class_names)], np.int32)
    
    i1 = 0
    for batch in range(len(testList)//batchSize):
        i2 = min(batchSize * (batch+1), len(testList))
        pathList = []
        for test in testList[i1: i2]:
            pathList.append(os.path.join(path, test))
        imgs = load_img(pathList, imgSize)
        
        bboxes, obj_probs, class_probs = sess.run(decode_op, feed_dict = {x: imgs, is_training: False})
        
        # [:, num_obj, 6]
        results = post_process(class_probs, obj_probs, bboxes, batchSize)
        for i, img_name in enumerate(testList[i1: i2]):
            img_info = imgInfo[img_name].copy()
            for gr_obj in img_info[:-1]:
                obj_gr_count[gr_obj[0]]+= 1
            pre = results[i]
            for pre_obj in pre:
                class_pre = int(pre_obj[0]) #class
                bbox_pre = [pre_obj[2]*img_info[-1][0], pre_obj[3]*img_info[-1][1], pre_obj[4]*img_info[-1][0], pre_obj[5]*img_info[-1][1]]
                conf_pre = pre_obj[1]
                iou_log = []
                for j, gr_obj in enumerate(img_info[:-1]):
                    class_gr = gr_obj[0]
                    bbox_gr = [gr_obj[1]['xmin'], gr_obj[1]['ymin'], gr_obj[1]['xmax'], gr_obj[1]['ymax']]
                    if class_pre == class_gr:
                        iou = cal_iou(bbox_gr, bbox_pre)
                        if iou > 0.5:
                            iou_log.append([j, iou])
                if len(iou_log) > 0:
                    iou_log = np.array(iou_log)
                    j = int(iou_log[np.argmax(iou_log[:, 1]), 0])
                    img_info.remove(img_info[j])
                    count[class_pre].append([conf_pre, 1, 0])
                else:
                    count[class_pre].append([conf_pre, 0, 1])
        i1 = i2
        if batch % 10 == 0:
            print(batch)
                
    precision_recall = []        
    for class_name, class_info in enumerate(count):
        if class_info != []:
            class_info = np.array(class_info, np.float32)
            seq = np.argsort(-class_info[:, 0], axis=0)
            class_info = class_info[seq]
            record = np.zeros([len(class_info), 4], np.float32)
            record[0, 0] = class_info[0, 1]
            record[0, 1] = class_info[0, 2]
            for i in range(len(record) - 1):
                record[i+1, :2] = record[i, :2] + class_info[i+1, 1:3]
            record[:, 2] = record[:, 0] / (record[:, 0] + record[:, 1])
            record[:, 3] = record[:, 0] / (obj_gr_count[class_name] + 1e-6)
            record = record[:, 2:4]
            
            i = 0
            while (i < len(record) - 1):
                if abs(record[i, 1] - record[i+1, 1]) < 1e-2:
                    record[i, 0] = max(record[i, 0], record[i+1, 0])
                    record = np.delete(record, i+1, 0)
                else:
                    i+= 1
            precision_recall.append(record)
        else:
            precision_recall.append(np.array([[0,0]], np.float32))
    AP = []
    for class_info in precision_recall:
        area = np.zeros([len(class_info)+1], np.float32)
        for i in range(len(class_info)):
            if i == 0:
                area[i] = max(class_info[i:, 0]) * class_info[i, 1]
            else:
                area[i] = (class_info[i, 1] - class_info[i-1, 1]) * max(class_info[i:, 0])
        AP.append(np.sum(area))
    mAP = np.mean(np.array(AP))
    
    t2 = time.time()
    print('time: %f\nmAP: %f' %(t2 - t1, mAP))
    return AP, mAP

if __name__ == '__main__' :
    tf.reset_default_graph()
    class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
                   'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 
                   'sofa', 'train', 'tvmonitor']
    imgInfo = np.load('./data/dataset.npy').item()
#    testList = np.load('./data/testList.npy').item()[1]
    testList = np.load('./data/trainList.npy').item()[1][:1000]
    path = np.load('./data/path.npy').item()[1]
    
    imgSize = 416  
    anchors = np.array([33.3821, 44.4139, 59.3492, 115.195, 159.471, 117.935, 87.7129, 208.548, 209.227, 227.445, 126.862, 320.245, 352.677, 202.811, 243.887, 357.922, 374.698, 370.688]).reshape([9, 2]).astype(np.float32)
    batchSize = 5
    
    x = tf.placeholder(tf.float32, [None, imgSize, imgSize, 3])
    is_training = tf.placeholder(tf.bool)
    exp_name = 'exp2'
    sess = tf.Session()
    
    out = build_net(x, is_training)
    v_list = tf.contrib.framework.get_variables_to_restore()
    saver = tf.train.Saver(v_list)
    file = tf.train.latest_checkpoint('./Weights/%s/fst_stage/' %exp_name)
    saver.restore(sess, file)
    
    decode_op = decode(out, imgSize, 20, anchors)
    
    AP, mAP = get_mAP(sess, x, decode_op, is_training, out, class_names, imgInfo, testList, path, imgSize, anchors, batchSize)
