# -*- coding: utf-8 -*-
from __future__ import division
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
import sys,os
import csv
import caffe
import pandas as pd
caffe.set_mode_gpu()
net = caffe.Net('/media/twj/Ubuntu/he/plant1/roc/test(bn).prototxt',1,
                weights='/media/twj/Ubuntu/he/plant/snapshots(256)/res__iter_30000.caffemodel')
#根据测试prototxt里面的批量大小设置
test_batch = 50
#根据测试集里面图片总数设置
test_num = 3544.0
test_N = int(np.ceil(test_num/test_batch))
number_of_files_processed = 0
label = []
#根据类别总数设置
score =np.zeros((1,12))
yuce_label= []
for test_it in range(test_N):
    net.forward()
    number_of_files_processed += 1
    prob= net.blobs['prob'].data
    
    yuce = net.blobs['prob'].data.argmax(1)
    #print prob.shape
    score = np.vstack((score, prob))
    #print score.shape
    
    label1= net.blobs['label'].data
    
    label = np.append(label, label1)
    label =np.int32(label[0:3544])
    
    yuce_label = np.append(yuce_label, yuce)
    yuce_label =np.int32(yuce_label[0:3544])
'''    
gt=np.load('/media/twj/Ubuntu/he/plant1/roc/label.npy')
yc=np.load('/media/twj/Ubuntu/he/plant1/roc/yuce_label.npy')
'''
gt=label
yc=yuce_label
T=0
SUM =0

labels = open("./labels.txt","r").readlines()
labels = [x.strip() for x in labels]
#根据类别总数设置
for j in range(12):
    for i in range(3544):
        a= gt[i]
        b= yc[i]
    #print a,b
        if(a==j):
            SUM =SUM +1
            #print gt[i]
        #print SUM
            if(a==b):
                T=T+1
    acc=T/SUM
    print acc
f1=f1_score(gt, yc, average=None)

cm = confusion_matrix(gt, yc)
print np.mean(f1)
print cm

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
ax = sns.heatmap(cm_normalized, xticklabels = labels, cmap=cmap, yticklabels = labels, linewidths=0.4,annot=True)
plt.yticks(rotation=0, fontsize=13)
plt.xticks(rotation=0, fontsize=13)
plt.xlabel('Predict classes',fontsize=16)
plt.ylabel('Actual classes',fontsize=16)
try:
    os.mkdir("./confusion_matrices")
except:
    pass


fig = plt.gcf()
fig.set_size_inches(15, 9)
plt.title("Confusion Matrix")
fig.savefig("./confusion_matrices/"+"_confusion_matrix_"+".png")
plt.clf()
