# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:23:49 2015

@author: shizenglin
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio
# Make sure that caffe is on the python path:
import sys
caffe_root = '/home/peiyong/Work/Zenglin/caffe-szl/'
sys.path.insert(0,caffe_root+'python')
import caffe
import cv2

caffe.set_mode_cpu()
minIternum=0;
maxIter=282240;
for iternum in xrange(282240,maxIter+1,10080):
    net = caffe.Net(caffe_root + 'examples/crowd/code/shanghaiA/deploy1.prototxt',
                    caffe_root + 'examples/crowd/code/shanghaiA/model/network/network_v3_ncl_iter_1053000.caffemodel',
                    caffe.TEST)
                    
    dataset = 'ShanghaiTech/partA'
    imPath = '/home/peiyong/Work/Zenglin/caffe-szl/examples/crowd/data/'+dataset+'/test/img/'
    dmapPath = '/home/peiyong/Work/Zenglin/caffe-szl/examples/crowd/data/'+dataset+'/test/dmap4/'
    testset=range(1,183)
    for idx in xrange(2,3):
    	    print(idx)
    	    imName = imPath+'IMG_'+str(idx)+'.jpg'
    	    dmapName = dmapPath +'DMAP_' + str(idx)+'.mat'
    
    	    dmap = sio.loadmat(dmapName)
    	    densitymap = dmap['dmap']
            print densitymap.sum()
    
    	    imArr = np.array(Image.open(imName).convert('L'))
    	    imShape = imArr.shape
    	    imHeight = imShape[0]
    	    imWidth = imShape[1]
    
    	    net.blobs['data'].reshape(1, 1, imHeight, imWidth)
                 
            net.blobs['data'].data[...] = imArr
            out = net.forward()
            p_dmap = net.blobs['avgscore'].data
            print p_dmap.sum()
            #print 'conv1'
            #print net.blobs['avgscore'].data
            #print 'conv4-11'
            #print net.blobs['conv4_11'].data
            #print 'conv4-13'
            #print net.blobs['conv4_13'].data
            #print 'conv4'
            #print net.blobs['conv4'].data
            #print 'convscore'
            #print net.blobs['conv_score_pre'].data
            #print 'w'
            #print net.params['conv_pre'][0].data
            #print 'b'
            #print net.params['conv_pre'][1].data
            #print p_dmap.sum()
            p_dmap=p_dmap.reshape(p_dmap.shape[2], p_dmap.shape[3])
    
            imdmap = Image.fromarray(densitymap*255)
            imdmap.show()
            cv2.imwrite('/home/peiyong/Work/Zenglin/caffe-szl/examples/crowd/code/shanghaiA/gt.jpg',densitymap*255)
            #imdmap.save("dmap.png")
            p_max=p_dmap.max()
            p_min=p_dmap.min()
            p_dmap = (p_dmap - p_min)/(p_max - p_min)
            #print p_dmap.sum()
            imp_dmap = Image.fromarray(p_dmap*255)
           
            imp_dmap.show()
            cv2.imwrite('/home/peiyong/Work/Zenglin/caffe-szl/examples/crowd/code/shanghaiA/predict.jpg',p_dmap*255)
            #imp_dmap.save("p_dmap.png")
