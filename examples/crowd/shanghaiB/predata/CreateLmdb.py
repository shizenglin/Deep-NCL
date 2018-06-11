# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 00:00:04 2015
@author: zhangyingying
Nine subimages are croped from each iamge.

"""
import sys
sys.path.append('..')

import numpy as np
import writelmdb2 as wlb

dataset = 'ShanghaiTech/partB'
downscale = 8.0
cropscale = 0.8
trainset=range(1,401)#np.random.permutation(range(1,401))
testset=range(1,317)

for i in [281,303,392,395]:
    trainset.remove(i)
for i in [262]:
    testset.remove(i)

trainImagePath = '/home/peiyong/Work/Zenglin/caffe-szl/examples/crowd/data/'+dataset+'/train/img/IMG_'
trainDmapPath = '/home/peiyong/Work/Zenglin/caffe-szl/examples/crowd/data/'+dataset+'/train/dmap8_5/DMAP_'
testImagePath = '/home/peiyong/Work/Zenglin/caffe-szl/examples/crowd/data/'+dataset+'/test/img/IMG_'
testDmapPath = '/home/peiyong/Work/Zenglin/caffe-szl/examples/crowd/data/'+dataset+'/test/dmap8_5/DMAP_'
trainLmdbPath = '/home/peiyong/Work/Zenglin/szl/'+dataset+'/train8_5_0.8'
testLmdbPath = '/home/peiyong/Work/Zenglin/szl/'+dataset+'/test8_5_0.8'

wlb.WriteLmdbTrain(trainImagePath,trainDmapPath,trainLmdbPath,trainset,downscale,True,cropscale)
wlb.WriteLmdbTestV2(testImagePath,testDmapPath,testLmdbPath,testset,downscale,False)

