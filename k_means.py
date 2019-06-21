# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:50:43 2019

@author: cfd_Liu
"""

import numpy as np

anchors = np.array([10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326], np.float32).reshape([9,2]) / 416
imgInfo = np.load('./data/dataset.npy').item()
size = []
for info in imgInfo.values():
    W = info[-1][0]
    H = info[-1][1]
    info.remove([W, H])
    if info != []:
        for obj in info:
            xy = obj[1]
            w = (xy['xmax'] - xy['xmin']) / W
            h = (xy['ymax'] - xy['ymin']) / H
            size.append([w,h])
size = np.array(size, np.float32)

for epoch in range(200):
    cluster = {i:[] for i in range(9)}
    for i, wh in enumerate(size):
        dist = np.zeros(9)
        for j, wh_an in enumerate(anchors):
            dist[j] = (wh[0] - wh_an[0])**2 + (wh[1] - wh_an[1])**2
        cluster[np.argmin(dist)].append(i)
        
    anchors_n = []
    for i in range(9):
        w, h = np.mean(size[cluster[i]], axis = 0)
        anchors_n.append([w, h])
    anchors_n = np.array(anchors_n, np.float32)
    err = np.max(np.mean(((anchors_n - anchors) * 416)**2, axis = 1))
    anchors = anchors_n.copy()
    if epoch%2==0:
        print(epoch, err)
    if err<1:
        break
anchors*=416
