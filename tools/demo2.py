#!/usr/bin/env python
# -*- coding: UTF-8 -*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import matplotlib
matplotlib.use('Agg')
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
from matplotlib.font_manager import FontProperties
zhfont1 = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

import csv
import time

from nets.vgg16 import vgg16
from nets.resnet_v1_rfcn_hole import resnetv1
import sys
reload(sys)
sys.setdefaultencoding('utf8')


CLASSES = ('__background__',
           'dr0', 'dr1', 'dr2', 'dr3')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_200000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}
localtime = time.asctime( time.localtime(time.time()) )

#thresh在demo()设置了CONF_THRESH=0.8只有概率大于0.8才会显示
#inds被保留的区块序号，区块信息在dets里
def vis_detections(im, class_name, dets, thresh=0.5 ,image_name='null'):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        tm=[0,0,2,2]
        return 0,0,tm,thresh
    #print(inds)
    #im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    temp=0
    maxscore = max(dets[:, -1])
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score==maxscore:
            return class_name,score,bbox,thresh
        #ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),bbox[2] - bbox[0],bbox[3] - bbox[1], fill=False,edgecolor='red', linewidth=3.5))
        #ax.text(bbox[0], bbox[1] - 2,'{:s} {:.3f}'.format(class_name, score),bbox=dict(facecolor='blue', alpha=0.5),fontsize=14, color='white')

    #ax.set_title(('{} detection results p({} | box) >= {:.1f}').format(class_name, class_name,thresh),fontsize=14)
    #plt.axis('off')
    #plt.tight_layout()
    #plt.draw()
    #plt.savefig("/var/www/html/figure/"+image_name)

def vis_detections_onlyone(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    maxscore = max(dets[:, -1])
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score==maxscore:
            ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),bbox[2] - bbox[0],bbox[3] - bbox[1], fill=False,edgecolor='red', linewidth=3.5))
            ax.text(bbox[0], bbox[1] - 2,'{:s} {:.3f}'.format(class_name, score),bbox=dict(facecolor='blue', alpha=0.5),fontsize=14, color='white')

    ax.set_title(('{} detection results '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    # im_dect at test.py
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    #print(scores)
    #print(boxes)
    timer.toc()
    print('检测区域采样时间 {:.3f}s 共计 {:d} 个目标区块'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    #enumerate枚举 hstack矩阵拼接 keep记录nms筛选后的区块 dets保存的每个区块的（x1 y1 x2 y2 score)格式list
    b=0
    e='error'
    f=[0,0,2,2]
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        #print(keep)
        dets = dets[keep, :]
        a=image_name
        c,d,g,t=vis_detections(im, cls, dets, thresh=CONF_THRESH, image_name=a)
        if (c==0 and d==0):
            1+1
        else:
            if (d>b):
                b=d
                e=c
                f=g
    if (e=='dr0'):
        e='正常人dr0'
    if (e=='dr1'):
        e='轻度患者dr1'
    if (e=='dr2'):
        e='中度患者dr2'
    if (e=='dr3'):
        e='重度患者dr3'
    if (e=='dr4'):
        e='增殖患者dr4'
    print(e,b)
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(im, aspect='equal')
    ax.add_patch(plt.Rectangle((f[0], f[1]),f[2] - f[0],f[3] - f[1], fill=False,edgecolor='red', linewidth=3.5))
    ax.text(f[0], f[1] + 25,'{:s} {:.3f}'.format(e, b),bbox=dict(facecolor='blue', alpha=0.5),fontsize=14, color='white',fontproperties=zhfont1)

    ax.set_title(('辅助诊断结果:{} p({} | box) >= {:.1f}').format(e, e,t),fontsize=14,fontproperties=zhfont1)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    #plt.savefig("/var/www/html/figure/"+image_name)
    plt.savefig("figure/"+localtime+image_name)
    filer=open('results/result'+localtime,'a+')
    filer.write(image_name+' '+e+' '+str(b)+'\n')
    filer.close

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args

def csv_writer(data, filename):
    with open(filename, "wb") as csv_file:
        writer = csv.writer(csv_file)
        for line in data:
            writer.writerow(line)

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    print ("\033[1;34mThe code is licensed by engineer1109.")
    #print ("\033[1;34m开始初始化系统")
    # init session
    sess = tf.Session(config=tfconfig)
    #print ("\033[1;33m卷积模型开始加载，默认是RES101")
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
        net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError
    #print(demonet)
    net.create_architecture(sess, "TEST", 5,
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    #print(saver.restore(sess, tfmodel))

    #print('网络加载完毕 {:s}'.format(tfmodel))
    result=[]
    fd = file("images.txt", "r" )  
    for line in fd.readlines():  
        result.append(list(map(str,line.split(','))))  
    #print (result)
    print ("欢迎使用TF-RFCN测试模式</br>")
    print ("总共需要辨识的图片数量</br>")
    size=len(result)
    print (size)
    image_name = [1]*size
    for i in range(size):
        var=str(result[i][0])
        var=var.strip()
        image_name[i]=var
    #print (image_name)
    #print(type(result))
    for image_name in image_name:
        print('</br>=======================</br>')
        print('====测试-TF-RFCN====</br>')
        print('测试数据 data/demo/{}</br>'.format(image_name))
        demo(sess, net, image_name)
        data = []
    data2=[]
    with open('results/result'+localtime) as f:
        for line in f:
            data2.append(line.strip().split(" "))
    #print('</br>概率分布</br>')
    #print (data2)
    filename = "csvfiles/output"+localtime+".csv"
    csv_writer(data2, filename)
    filename = "output.csv"
    csv_writer(data2, filename)
    #plt.show()
