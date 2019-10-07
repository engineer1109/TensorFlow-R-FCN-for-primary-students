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
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_150000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}
LANGUAGE= {'en': ('en',),'zh': ('zh',)}

localtime = time.asctime( time.localtime(time.time()) )

def vis_detections(im, class_name, dets, thresh=0.8 ,image_name='null'):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        tm=[0,0,2,2]
        return 0,0,tm,0
    #print(inds)
    #im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    temp=0
    bbox=list(range(len(inds)))
    score=list(range(len(inds)))
    clsa=list(range(len(inds)))
    #maxscore = max(dets[:, -1])
    for i in inds:
        bbox[temp] = dets[i, :4]
        score[temp] = dets[i, -1]
        clsa[temp]=class_name
        temp=temp+1
    return clsa,score,bbox,temp
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
    print('Dectecting Time {:.3f}s'.format(timer.total_time))

    # Visualize detections for each class
    CONF_THRESH = 0.7
    NMS_THRESH = 0.3

    b=0
    e='error'
    f=[0,0,2,2]
    y=0
    sctext='0'
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(im, aspect='equal')

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
        clsa,score,bbox,t=vis_detections(im, cls, dets, thresh=CONF_THRESH, image_name=a) 
        if(t==0):
            1+1
        else:
            for u in range(t):   
                ax.add_patch(plt.Rectangle((bbox[u][0], bbox[u][1]),bbox[u][2]-bbox[u][0],bbox[u][3]-bbox[u][1], fill=False,edgecolor='red', linewidth=3.5))
                ax.text(bbox[u][0], bbox[u][1] - 25,'{:s} {:.3f}'.format(cls, score[u]),bbox=dict(facecolor='blue', alpha=0.5),fontsize=14, color='white',fontproperties=zhfont1)
            y=y+1

    if(y==0):
        ax.text(25, 25,'Nothing Detected',bbox=dict(facecolor='blue', alpha=0.5),fontsize=14, color='white',fontproperties=zhfont1)

    ax.set_title(('Detecting Result'),fontsize=14,fontproperties=zhfont1)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig("figure/"+image_name)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    parser.add_argument('--language', dest='language', help='Language choice',
                        choices=LANGUAGE.keys(),default='en')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    lang=args.language
    if(lang=='en'):
        print('English Language Mode')
    elif(lang=='zh'):
        print('中文模式')
        CLASSES = ('__background__',
           '飞机', '自行车', '鸟', '船',
           '瓶子', '巴士', '汽车', '猫', '椅子',
           '牛', '餐桌', '狗', '马',
           '摩托车', '人', '盆栽',
           '羊', '沙发', '火车', '电视')
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    print ("\033[1;34mThe code is licensed by engineer1109.")

    # init session
    sess = tf.Session(config=tfconfig)

    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
        net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError

    net.create_architecture(sess, "TEST", len(CLASSES),
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    files=[]
    for _,_,demofiles in os.walk(r"./data/demo"):
        files=demofiles

    print ("tf-rfcn Test")

    for image_name in files:
        print('Image data/demo/{}'.format(image_name))
        demo(sess, net, image_name)
        data = []
