# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:01:44 2019

@author: cfd_Liu
"""

import tensorflow as tf
import numpy as np
import random
import os
import cv2

def div_train_test():
    path = os.path.abspath(r'C:\Users\cfd_Liu\Desktop\Machine Learning\Code\PracticeCode\TensorFlow Learning\OpenCV\LaneDet\YOLO\yolo_train\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages')
    dirlist = os.listdir(path)
    testList = []
    trainList = dirlist.copy()
    randint = random.sample(range(0, len(dirlist)), 1000)
    for i in randint:
        name = dirlist[i]
        testList.append(name)
    for i in range(len(randint)):
        trainList.remove(testList[i])
    np.save('./data/trainList', {1:trainList})
    np.save('./data/testList', {1:testList})
    np.save('./data/path', {1:path})

def conv2d(x, outfilters, name, name_ind, is_training, size=3, stride=1, batchnorm=True, trainable=False):
    def fixed_pad(x, stride=1):
        if stride>1:
            x = tf.pad(x, [[0,0], [1,1], [1,1], [0,0]]) #we only need to pad one pixel, the last row and column pad has no effect
        return x
    
    x = fixed_pad(x, stride)
    
    regularizer = tf.contrib.layers.l2_regularizer(5e-4) if name!='conv' else None
    w_initializer = tf.contrib.layers.xavier_initializer()
    b_initializer = tf.constant_initializer(0.)
    if batchnorm == True:
        xx = tf.layers.conv2d(x, outfilters, size, stride, padding = 'SAME' if stride==1 else 'VALID', 
                              use_bias=False, kernel_initializer=w_initializer, 
                              kernel_regularizer=regularizer, trainable=trainable, name=name + str(name_ind))
        xx = tf.contrib.layers.batch_norm(xx, scale=True, 
                                          is_training = is_training, trainable = trainable)
        return tf.nn.leaky_relu(xx, 0.1), name_ind + 1
    else:
        xx = tf.layers.conv2d(x, outfilters, size, stride, padding = 'SAME' if stride==1 else 'VALID', 
                              kernel_initializer=w_initializer, bias_initializer=b_initializer, 
                              kernel_regularizer=regularizer, trainable=trainable, name=name + str(name_ind))
        return xx, name_ind + 1
    
def route(x1, x2):
    [_, H, W, _]= x1.get_shape().as_list()
    x1 = tf.image.resize_nearest_neighbor(x1, [H*2, W*2])
    return tf.concat([x1, x2], axis=3)

def darknet53_block(x, outfilters, name, name_ind, is_training, trainable = False):
    shortcut = x
    x, name_ind = conv2d(x, outfilters, name, name_ind, is_training, size=1, trainable = trainable)
    x, name_ind = conv2d(x, outfilters*2, name, name_ind, is_training, trainable = trainable)
    return x + shortcut, name_ind

def yolo_block(x, outfilters, name, name_ind, is_training, trainable = True, num_class=20):
    x, name_ind = conv2d(x, outfilters, name, name_ind, is_training, size=1, trainable = trainable)
    x, name_ind = conv2d(x, outfilters*2, name, name_ind, is_training, trainable = trainable)
    x, name_ind = conv2d(x, outfilters, name, name_ind, is_training, size=1, trainable = trainable)
    x, name_ind = conv2d(x, outfilters*2, name, name_ind, is_training, trainable = trainable)
    x, name_ind = conv2d(x, outfilters, name, name_ind, is_training, size=1, trainable = trainable)
    route = x
    x, name_ind = conv2d(x, outfilters*2, name, name_ind, is_training, trainable = trainable)
    x, name_ind = conv2d(x, 3*(5+num_class), name, name_ind, is_training, size=1, batchnorm=False, trainable = trainable)
    return x, route
def build_darknet(x, is_training, trainable=True):
    'build darknet53'
    net, name_ind = conv2d(x, 32, 'conv', 0, is_training, trainable = trainable)
    net, name_ind = conv2d(net, 64, 'conv', name_ind, is_training, stride=2, trainable = trainable)
    net, name_ind = darknet53_block(net, 32, 'conv', name_ind, is_training, trainable = trainable)
    net, name_ind = conv2d(net, 128, 'conv', name_ind, is_training, stride=2, trainable = trainable)
    
    for i in range(2):
        net, name_ind = darknet53_block(net, 64, 'conv', name_ind, is_training, trainable = trainable)
    net, name_ind = conv2d(net, 256, 'conv', name_ind, is_training, stride=2, trainable = trainable)
    for i in range(8):
        net, name_ind = darknet53_block(net, 128, 'conv', name_ind, is_training, trainable = trainable)
        
    route1 = net
    net, name_ind = conv2d(net, 512, 'conv', name_ind, is_training, stride=2, trainable = trainable)
    for i in range(8):
        net, name_ind = darknet53_block(net, 256, 'conv', name_ind, is_training, trainable = trainable)
        
    route2 = net
    net, name_ind = conv2d(net, 1024, 'conv', name_ind, is_training, stride=2, trainable = trainable)
    for i in range(4):
        net, name_ind = darknet53_block(net, 512, 'conv', name_ind, is_training, trainable = trainable)
    return net, route1, route2

def build_detnet(net, route1, route2, is_training, trainable=True):
    out = []
    #det1
    out1, route3 = yolo_block(net, 512, 'det1_', 0, is_training, trainable = trainable)
    out.append(out1)
    
    #det2
    net, name_ind = conv2d(route3, 256, 'det2_', 0, is_training, size=1, trainable = trainable)
    net = route(net, route2)
    out2, route4 = yolo_block(net, 256, 'det2_', name_ind, is_training, trainable = trainable)
    out.append(out2)
    
    #det3
    net, name_ind = conv2d(route4, 128, 'det3_', 0, is_training, size=1, trainable = trainable)
    net = route(net, route1)
    out3, _ = yolo_block(net, 128, 'det3_', name_ind, is_training, trainable = trainable)
    out.append(out3)
    return out

def proc_anchors(anchors, imgSize):
    anchorCoor = [np.zeros([13, 13, 3, 4], np.float32), np.zeros([26, 26, 3, 4], np.float32), np.zeros([52, 52, 3, 4], np.float32)]
    anchor = [anchors[6:9, :] / imgSize, anchors[3:6, :] / imgSize, anchors[:3, :] / imgSize]
    for n in range(3):
        cellNum = 13 if n==0 else n * 26
        for i in range(cellNum):
            for j in range(cellNum):
                for k in range(3):
                    y_cen = 1/cellNum * i + 1/cellNum/2
                    x_cen = 1/cellNum * j + 1/cellNum/2
                    xmin = x_cen - anchor[n][k, 0]/2
                    ymin = y_cen - anchor[n][k, 1]/2
                    xmax = x_cen + anchor[n][k, 0]/2
                    ymax = y_cen + anchor[n][k, 1]/2
                    anchorCoor[n][i, j, k] = [xmin, ymin, xmax, ymax]#after normalization
    return anchorCoor

def cal_iou(bbox1, bbox2):
    xi1 = max(bbox1[0], bbox2[0])
    yi1 = max(bbox1[1], bbox2[1])
    xi2 = min(bbox1[2], bbox2[2])
    yi2 = min(bbox1[3], bbox2[3])
    int_area = (max(yi2, 0) - max(yi1, 0)) * (max(xi2, 0) - max(xi1, 0))
    bbox1_area = (bbox1[3]-bbox1[1])*(bbox1[2]-bbox1[0])
    bbox2_area = (bbox2[3]-bbox2[1])*(bbox2[2]-bbox2[0])
    uni_area = bbox1_area + bbox2_area - int_area
    iou = int_area/ uni_area
    return iou

def cal_wh(bbox_gr, bbox_pri):
    w = (bbox_gr[2] - bbox_gr[0])/(bbox_pri[2] - bbox_pri[0])
    h = (bbox_gr[3] - bbox_gr[1])/(bbox_pri[3] - bbox_pri[1])
    return np.log(w), np.log(h)

def proc_bbox_batch(bbox_batch):
    maxnum = 0
    for bbox_img in bbox_batch:
        maxnum = len(bbox_img) if len(bbox_img)>maxnum else maxnum
    bbox = np.zeros([len(bbox_batch), maxnum, 4], np.float32)
    for i, bbox_img in enumerate(bbox_batch):
        for j, each_bbox in enumerate(bbox_img):
            bbox[i,j] = np.array(each_bbox, np.float32)
    return bbox #[x_cen, y_cen, w, h] in range (0,1)

def cal_y(batchList, imgInfo, anchorCoor):
    batchSize = len(batchList)
    y = [np.zeros([batchSize, 13, 13, 3, 25], np.float32), np.zeros([batchSize, 26, 26, 3, 25], np.float32), np.zeros([batchSize, 52, 52, 3, 25], np.float32)]
    bbox_batch = []
    for img_ind, img in enumerate(batchList):
        Info = imgInfo[img].copy()
        w, h = Info[-1]
        Info.remove([w, h])
        bbox_img = []
        for objInfo in Info:
            iou_log = np.zeros([3, 3, 9], np.float32)
            class_ind = objInfo[0]
            bbox_gr = [objInfo[1]['xmin']/w, objInfo[1]['ymin']/h, objInfo[1]['xmax']/w, objInfo[1]['ymax']/h] # in range (0, 1)
            bbox_img.append([(bbox_gr[2]+bbox_gr[0])/2, (bbox_gr[3]+bbox_gr[1])/2, (bbox_gr[2]-bbox_gr[0]), (bbox_gr[3]-bbox_gr[1])])
            x_cen_gr = (bbox_gr[0] + bbox_gr[2]) / 2.
            y_cen_gr = (bbox_gr[1] + bbox_gr[3]) / 2.
            for n in range(3):
                cellNum = 13 if n==0 else n * 26
                x_cell_ind = int(x_cen_gr * cellNum // 1)
                y_cell_ind = int(y_cen_gr * cellNum // 1)
                x_cen = x_cen_gr * cellNum - x_cell_ind
                y_cen = y_cen_gr * cellNum - y_cell_ind
                for anch in range(3):
                    bbox_pri = anchorCoor[n][y_cell_ind, x_cell_ind, anch]
                    iou = cal_iou(bbox_pri, bbox_gr)
                    w_gr, h_gr = cal_wh(bbox_gr, bbox_pri)
                    iou_log[n, anch] = [iou, x_cell_ind, y_cell_ind, x_cen, y_cen, w_gr, h_gr, x_cen_gr, y_cen_gr]
                    
            p1 = 0; p2 = 0; t = iou_log[0,0,0]
            for i in range(3):
                for j in range(3):
                    if iou_log[i, j, 0] > t:
                        p1 = i; p2 = j; t = iou_log[i, j, 0]
                        
            y[p1][img_ind, int(iou_log[p1, p2, 2]), int(iou_log[p1, p2, 1]), p2, class_ind + 5] = 1
            y[p1][img_ind, int(iou_log[p1, p2, 2]), int(iou_log[p1, p2, 1]), p2, 4] = 1
            y[p1][img_ind, int(iou_log[p1, p2, 2]), int(iou_log[p1, p2, 1]), p2, 0] = iou_log[p1, p2, 3]
            y[p1][img_ind, int(iou_log[p1, p2, 2]), int(iou_log[p1, p2, 1]), p2, 1] = iou_log[p1, p2, 4]
            y[p1][img_ind, int(iou_log[p1, p2, 2]), int(iou_log[p1, p2, 1]), p2, 2] = iou_log[p1, p2, 5]
            y[p1][img_ind, int(iou_log[p1, p2, 2]), int(iou_log[p1, p2, 1]), p2, 3] = iou_log[p1, p2, 6]
        bbox_batch.append(bbox_img)
    return y, proc_bbox_batch(bbox_batch)
    
def cal_loss(out_pre, out_gr, anchorCoor, bbox, focal_loss=False, regularization=False):
    #[2][none, 13, 13, 75]
    #[2][None, 13, 13, 3, 25]    
        
    loss = tf.constant(0, tf.float32)
    for n in range(3):
        cellNum = 13 if n==0 else n * 26
        y_pre = tf.reshape(out_pre[n], [-1, cellNum, cellNum, 3, 25])
        y_gr = out_gr[n]
        #[N, 13, 13, 3]
        mask = y_gr[:,:,:,:, 4]
        
        mask_coor = tf.stack([mask, mask], axis = 4)
        coor_xy_pre = tf.sigmoid(y_pre[:,:,:,:, :2])
        coor_xy_gr = y_gr[:,:,:,:, :2]
        coor_wh_pre = y_pre[:,:,:,:, 2:4]
        coor_wh_gr = y_gr[:,:,:,:, 2:4]
        coor_scale_mod = (2 - tf.exp(coor_wh_gr[..., 0:1]) * 
                          (anchorCoor[n][..., 2:3]-anchorCoor[n][..., :1]) * tf.exp(coor_wh_gr[..., 1:2]) * 
                          (anchorCoor[n][..., 3:4] - anchorCoor[n][..., 1:2]))
        coor_xy_loss = tf.reduce_sum(tf.reduce_mean(tf.square(coor_xy_pre - coor_xy_gr) * mask_coor * coor_scale_mod, axis=0))
        coor_wh_loss = tf.reduce_sum(tf.reduce_mean(tf.square(coor_wh_pre - coor_wh_gr) * mask_coor * coor_scale_mod, axis=0))
        
        ignore_mask = cal_ignore_mask(y_pre[..., :4], bbox, anchorCoor[n], cellNum)
        mask_no_obj = (1 - mask) * ignore_mask
        obj_pre = y_pre[:,:,:,:, 4]
        obj_gr = y_gr[:,:,:,:, 4]

        if focal_loss:
            focal_obj = 0.25 * tf.pow((1 - tf.sigmoid(obj_pre)), 2)
            focal_no_obj = 0.75 * tf.pow(tf.sigmoid(obj_pre), 2)
        else:
            focal_obj = 1
            focal_no_obj = 1
        no_obj_loss = tf.reduce_sum(tf.reduce_mean(focal_no_obj * tf.nn.sigmoid_cross_entropy_with_logits(labels = obj_gr, logits = obj_pre) * mask_no_obj, axis=0))
        
        obj_loss = tf.reduce_sum(tf.reduce_mean(focal_obj * tf.nn.sigmoid_cross_entropy_with_logits(labels = obj_gr, logits = obj_pre) * mask, axis=0))
        
        mask_class = tf.stack([mask for i in range(20)], axis = 4)
        class_pre = y_pre[:,:,:,:, 5:]
        class_gr = y_gr[:,:,:,:, 5:]
        class_loss = tf.reduce_sum(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = class_gr, logits = class_pre) * mask_class, axis=0))
        
        loss = loss + coor_xy_loss + coor_wh_loss + no_obj_loss + obj_loss + class_loss
        
    l2_loss = tf.losses.get_regularization_loss() if regularization else tf.constant(0.)
    loss = loss + l2_loss
    return [loss, coor_xy_loss, coor_wh_loss, no_obj_loss, obj_loss, class_loss, l2_loss]

def cal_ignore_mask(bbox_pre, bbox_gr, anchor, cellNum):
    height_ind = tf.range(cellNum, dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(height_ind, height_ind)
    #[N,13,13,3]
    bbox_pre_x = (tf.sigmoid(bbox_pre[..., 0]) + x_offset[:,:,np.newaxis]) / cellNum
    bbox_pre_y = (tf.sigmoid(bbox_pre[..., 1]) + y_offset[:,:,np.newaxis]) / cellNum
    bbox_pre_w = tf.exp(bbox_pre[..., 2]) * (anchor[..., 2] - anchor[..., 0])
    bbox_pre_h = tf.exp(bbox_pre[..., 3]) * (anchor[..., 3] - anchor[..., 1])
    #[N,13,13,3,4]
    bbox_pre = tf.stack([bbox_pre_x, bbox_pre_y, bbox_pre_w, bbox_pre_h], axis=4)
    #[N,13,13,3,M,4]
    bbox_pre = bbox_pre[:,:,:,:, np.newaxis, :]
    bbox_gr = bbox_gr[:, np.newaxis, np.newaxis, np.newaxis, :, :]
    
    bbox_pre_area = bbox_pre[..., 2] * bbox_pre[..., 3]
    bbox_gr_area = bbox_gr[..., 2] * bbox_gr[..., 3]

    bbox_pre = tf.concat([bbox_pre[..., :2] - bbox_pre[..., 2:] * 0.5,
                        bbox_pre[..., :2] + bbox_pre[..., 2:] * 0.5], axis=-1)
    bbox_gr = tf.concat([bbox_gr[..., :2] - bbox_gr[..., 2:] * 0.5,
                        bbox_gr[..., :2] + bbox_gr[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(bbox_pre[..., :2], bbox_gr[..., :2])
    right_down = tf.minimum(bbox_pre[..., 2:], bbox_gr[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    #[N, 13, 13, 3, M]
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = bbox_pre_area + bbox_gr_area - inter_area
    iou = 1.0 * inter_area / union_area
    #[N, 13, 13, 3]
    max_iou = tf.reduce_max(iou, axis=-1)
    return tf.cast(max_iou < 0.5, tf.float32)

def decode(out, imgSize, num_classes=80, anchorTotal=None):
    bboxesTotal = []; obj_probsTotal = []; class_probsTotal = []
    for i, detection_feat in enumerate(out):
        _, H, W, _ = detection_feat.get_shape().as_list()
        anchors = anchorTotal[(2-i)*3:(2-i)*3+3, :]
        num_anchors = len(anchors)
        
        detetion_results = tf.reshape(detection_feat, [-1, H * W, num_anchors, num_classes + 5])
        
        bbox_xy = tf.nn.sigmoid(detetion_results[:, :, :, :2])
        bbox_wh = tf.exp(detetion_results[:, :, :, 2:4])
        obj_probs = tf.nn.sigmoid(detetion_results[:, :, :, 4])
        class_probs = tf.nn.sigmoid(detetion_results[:, :, :, 5:])
    
        anchors = tf.constant(anchors, dtype=tf.float32)
    
        height_ind = tf.range(H, dtype=tf.float32)
        width_ind = tf.range(W, dtype=tf.float32)
        
        x_offset, y_offset = tf.meshgrid(height_ind, width_ind)
        x_offset = tf.reshape(x_offset, [1, -1, 1])
        y_offset = tf.reshape(y_offset, [1, -1, 1])
        
        bbox_x = (bbox_xy[:, :, :, 0] + x_offset) / W
        bbox_y = (bbox_xy[:, :, :, 1] + y_offset) / H
        bbox_w = bbox_wh[:, :, :, 0] * anchors[:, 0] / imgSize * 0.5
        bbox_h = bbox_wh[:, :, :, 1] * anchors[:, 1] / imgSize * 0.5

        bboxes = tf.stack([bbox_x - bbox_w, bbox_y - bbox_h, bbox_x + bbox_w, bbox_y + bbox_h], axis=3)
        bboxesTotal.append(bboxes)
        obj_probsTotal.append(obj_probs)
        class_probsTotal.append(class_probs)
    bboxes = tf.concat(bboxesTotal, axis=1)
    obj_probs = tf.concat(obj_probsTotal, axis=1)
    class_probs = tf.concat(class_probsTotal, axis=1)
    return bboxes, obj_probs, class_probs

def post_process(sess, x, is_training, out, pathList, imgSize, num_classes, class_names, anchors, save_weights=False):
    def arg_max(x):
        xmax = 0
        id1 = 0
        id2 = 0
        w1, w2 = x.shape
        for i in range(w1):
            for j in range(w2):
                if x[i, j]>xmax:
                    xmax = x[i,j]
                    id1 = i
                    id2 = j
        return id1, id2
    imgTotal = []
    imgSrcTotal = []
    for path in pathList:
        imgSrc = cv2.imread(path, 1)
        img = np.float32(cv2.cvtColor(imgSrc, cv2.COLOR_BGR2RGB)/255)
        img = cv2.resize(img, (imgSize, imgSize))[np.newaxis, :,:,:]
        imgTotal.append(img)
        imgSrcTotal.append(imgSrc)
    imgTotal = np.concatenate(imgTotal, axis=0)
        
    bboxTotal, obj_probsTotal, class_probsTotal = sess.run(decode(out, imgSize, num_classes, anchorTotal=anchors), feed_dict={x:imgTotal, is_training:False})
    if save_weights:
        saver = tf.train.Saver()
        saver.save(sess, './Weights/weightsInit.ckpt')
    
    for i, imgSrc in enumerate(imgSrcTotal):
        H, W, _ = imgSrc.shape
        #[:, 3]; [:, 3, 4]
        class_name = np.argmax(class_probsTotal[i], axis = 2)
        class_probs = np.max(class_probsTotal[i], axis = 2)
        obj_probs = obj_probsTotal[i]
        bbox = bboxTotal[i]
        bbox[bbox>1] = 1
        bbox[bbox<0] = 0
        confidence = class_probs * obj_probs
        conf = confidence.copy()
        confidence[confidence<0.5] = 0 ######################################################################################
        
        det_indTotal = []
        while (np.max(confidence)!=0):
            id1, id2 = arg_max(confidence)
            bbox1_area = (bbox[id1, id2,3]-bbox[id1, id2,1])*(bbox[id1, id2,2]-bbox[id1, id2,0])
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
                    bbox2_area = (bbox[coor[0],coor[1],3]-bbox[coor[0],coor[1],1]) * (bbox[coor[0],coor[1],2]-bbox[coor[0],coor[1],0])
                    uni_area = bbox1_area + bbox2_area - int_area
                    iou = int_area/uni_area
                    if iou>0.5: ###########################################################################################
                        sign = 1
                        break
            if sign==0:
                det_indTotal.append([id1, id2]) 
            confidence[id1, id2] = 0
            
        depict = []
        for [id1, id2] in det_indTotal:
            x1,y1,x2,y2 = bbox[id1, id2]
            x1 = int(x1*W); x2 = int(x2*W); y1 = int(y1*H); y2 = int(y2*H)
            name = class_names[class_name[id1, id2]]
            depict.append([x1,y1,x2,y2])
            cv2.rectangle(imgSrc, (x1,y1), (x2,y2), (255,0,0), 2)
            
            y = y1-8 if y1-8>8 else y1+8
            cv2.putText(imgSrc, '%s %.2f' %(name, conf[id1, id2]), (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
        cv2.imshow('%d' %i, imgSrc)
        cv2.imwrite('./test_imgs/%d.jpg' %i, imgSrc, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
def build_net(x, is_training):
    with tf.variable_scope('body'):
        net, route1, route2 = build_darknet(x, is_training)
    with tf.variable_scope('head'):
        out = build_detnet(net, route1, route2, is_training)
    return out
def load_img(path, batchList, imgSize):
    imgbatch = []
    for filename in batchList:
        filepath = os.path.join(path, filename)
        img = cv2.resize(cv2.imread(filepath, 1), (imgSize, imgSize))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[np.newaxis,:,:,:].astype(np.float32) / 255
        imgbatch.append(img)
    imgbatch = np.concatenate(imgbatch, axis=0)
    return imgbatch
