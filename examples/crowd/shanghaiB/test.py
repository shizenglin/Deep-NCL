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
caffe_root = '/home/shizenglin/caffe-szl/'
sys.path.insert(0,caffe_root+'python')
import caffe

caffe.set_mode_cpu()
minMAE=10000
minRMSE=10000
minIternum=0;
maxIter=540000;
for iternum in xrange(10800,maxIter+1,10800):
    net = caffe.Net(caffe_root + 'examples/crowd/code/shanghai/deploy2.prototxt',
                    caffe_root + 'examples/crowd/code/shanghai/model/network3/network_iter_'+str(iternum)+'.caffemodel',
                    caffe.TEST)
                    
    string_ = str(iternum/10800)+'/'+str(maxIter/10800)
    sys.stdout.write("\r%s" % string_)
    sys.stdout.flush()
    print('\n')
                    
    dataset = 'ShanghaiTech/partA/test'
    imPath = '/home/shizenglin/caffe-szl/examples/crowd/data/'+dataset+'/img/IMG_'
    dmapPath = '/home/shizenglin/caffe-szl/examples/crowd/data/'+dataset+'/dmap4/DMAP_'
    testset=range(1,183)
    dmapNum=[]
    imNum=[]
    print('\n')    
    print(minMAE)
    print(minRMSE)
    print(minIternum)
    for idx in xrange(len(testset)):
    	    print(idx+1)
    	    imName = imPath+str(testset[idx])+'.jpg'
    	    dmapName = dmapPath + str(testset[idx])+'.mat'
    
    	    dmap = sio.loadmat(dmapName)
    	    densitymap = dmap['dmap']
    	    dmap_sum = densitymap.sum()
    
    	    imArr = np.array(Image.open(imName).convert('L'))
    	    imShape = imArr.shape
    	    imHeight = imShape[0]
    	    imWidth = imShape[1]
         
    	    net.blobs['data'].reshape(1, 1, imHeight, imWidth)                      
    	    net.blobs['data'].data[...] = imArr
    	    out = net.forward()
    	    im_sum = net.blobs['avgscore'].data.sum()
    	    dmapNum.append(dmap_sum)
    	    imNum.append(round(im_sum))
    sublist = map(lambda x: x[0]-x[1], zip(dmapNum, imNum))
    abslist = map(abs,sublist)
    
    MAE=sum(abslist)/len(testset)
    RMSE = math.sqrt(sum(map(lambda x: pow(x,2), sublist))/len(testset))
    
    if MAE<=minMAE and RMSE<=minRMSE:
        minMAE=MAE
        minRMSE=RMSE
        minIternum=iternum
print('\n')    
print(minMAE)
print(minRMSE)
print(minIternum)