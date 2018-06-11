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

caffe.set_mode_gpu()
net = caffe.Net(caffe_root + 'examples/crowd/code/shanghaiA/deploy1.prototxt',
            caffe_root + 'examples/crowd/code/shanghaiA/vgg64-result/64-0.0001/network_vgg_v1_iter_2019600.caffemodel',
            caffe.TEST)
            
dataset = 'ShanghaiTech/partA/test'
imPath = '/home/peiyong/Work/Zenglin/caffe-szl/examples/crowd/data/'+dataset+'/img/IMG_'
dmapPath = '/home/peiyong/Work/Zenglin/caffe-szl/examples/crowd/data/'+dataset+'/dmap8/DMAP_'
testset=range(1,183)#183
dmapNum=0
imNum=0
allNum=0

for idx in xrange(len(testset)):
    print(idx+1)
    imName = imPath+str(testset[idx])+'.jpg'
    dmapName = dmapPath + str(testset[idx])+'.mat'

    dmap = sio.loadmat(dmapName)
    densitymap = dmap['dmap']
    dmap_sum = densitymap#.sum()

    imArr = np.array(Image.open(imName).convert('L'))
    imArr = imArr[np.newaxis,:,:]
    imArr = np.tile(imArr,(3,1,1))
    imShape = imArr.shape
    imHeight = imShape[1]
    imWidth = imShape[2]
 
    net.blobs['data'].reshape(1, 3, imHeight, imWidth)                      
    net.blobs['data'].data[...] = imArr
    out = net.forward()
    im_sum_avg = net.blobs['avgscore'].data#.sum()
    im_sum1=0
    im_sum2=0
    for i in xrange(1,65):
    	subim_sum = net.blobs['score'+str(i)].data#.sum()
        temp1=(subim_sum-dmap_sum)*(subim_sum-dmap_sum)
        temp2=(subim_sum-im_sum_avg)*(subim_sum-im_sum_avg)
        im_sum1 = im_sum1+temp1.sum()
        im_sum2 = im_sum2+temp2.sum()
   
    dmapNum = dmapNum+im_sum1/64
    imNum = imNum+im_sum2/64
    temp3=(im_sum_avg-dmap_sum)*(im_sum_avg-dmap_sum)
    allNum = allNum+temp3.sum()

print dmapNum/182
print imNum/182
print allNum/182





"""dmapNum.append(dmap_sum)
imNum.append(round(im_sum))
sublist = map(lambda x: x[0]-x[1], zip(dmapNum, imNum))
abslist = map(abs,sublist)

MAE=sum(abslist)/len(testset)
RMSE = math.sqrt(sum(map(lambda x: pow(x,2), sublist))/len(testset))

print MAE
print RMSE"""
