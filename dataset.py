# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:07:37 2019

@author: cfd_Liu
"""

import os
import numpy as np

def proc_line(line):
    line = line.replace('\t', '0')
    line = line.strip()
    for i in ['<', '>']:
        line = line.replace(i, ' ')
    line = line.split()
    return line

def proc_names(class_names):
    class_lab = {}
    for i, name in enumerate(class_names):
        class_lab[name] = i
    return class_lab

def proc_dataset():
    class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
                   'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 
                   'sofa', 'train', 'tvmonitor']
    class_lab = proc_names(class_names)
    imgInfo = {}
    
    path = os.path.abspath(r'C:\Users\cfd_Liu\Desktop\Machine Learning\Code\PracticeCode\TensorFlow Learning\OpenCV\LaneDet\YOLO\yolo_train\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations')
    dirs = os.listdir(path)
    for ind, filename in enumerate(dirs):
        file = os.path.join(path, filename)
        info = open(file)
        filedata = info.readlines()
        imgName = filename.replace('.xml', '.jpg')
        
        imgInfo[imgName] = []
        lenth = len(filedata)
        i = 0
        line_list = []
        while i<lenth :
            line = filedata[i]
            line = proc_line(line)
            if len(line)<2:
                i+=1
                continue
            if line[0]=='00' and line[1]=='name' :
                line_list.append(line)
                i+=1
            elif line[0]=='00' and line[1]=='bndbox' :
                for j in range(1, 5):
                    line_list.append(proc_line(filedata[i+j]))
                lineInfo = [class_lab[line_list[0][2]], 
                            {line_list[1][1]:(float(line_list[1][2])-1), 
                             line_list[2][1]:(float(line_list[2][2])-1), 
                             line_list[3][1]:(float(line_list[3][2])-1), 
                             line_list[4][1]:(float(line_list[4][2])-1)}]
                imgInfo[imgName].append(lineInfo)
                line_list = []
                i+=5
            elif line[0]=='00' and line[1]=='width' :
                w = int(line[2])
                i+=1
            elif line[0]=='00' and line[1]=='height' :
                h = int(line[2])
                i+=1
            else:
                i+=1
        imgInfo[imgName].append([w, h])
    np.save('./dataset', imgInfo)

if __name__ == '__main__' :
#    proc_dataset()
    imgInfo = np.load('./dataset.npy').item()
