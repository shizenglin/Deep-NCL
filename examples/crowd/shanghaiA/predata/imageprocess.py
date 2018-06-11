# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 23:53:36 2015
@author: zhangyingying
"""

import math
import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import rescale
import time
import cv2
def ReadImage(imPath,mirror = False,scale=1.0):
    """
    Read gray images.
    """ 
    imArr = np.array(Image.open(imPath).convert('L'))
    #print imArr
    if(scale!=1):
        imArr = rescale(imArr, scale,preserve_range=True)#
        #print imArr
        #time.sleep(10)
    imArr = imArr[np.newaxis,:,:]
    imArr = np.tile(imArr,(3,1,1))
    if not mirror:
        return imArr
    else:
        imArr_m = np.zeros(imArr.shape,dtype=np.uint8)
        for i in xrange(imArr.shape[2]):
            imArr_m[:,:,i] = imArr[:,:,imArr.shape[2]-1-i]
        return imArr_m

def ReadDmap(dmapPath,mirror = False,scale = 1.0):
    """
    Load the density map from matfile.
    """ 
    dmap = sio.loadmat(dmapPath)
    densitymap = dmap['dmap']
    if(scale!=1):
        dmap_sum = densitymap.sum()
        densitymap = rescale(densitymap, scale, preserve_range=True)#
        res_sum = densitymap.sum()
        if res_sum != 0:
            densitymap= densitymap * (dmap_sum/res_sum)
    densitymap = densitymap.reshape(1,densitymap.shape[0],densitymap.shape[1])
    if not mirror:
        return densitymap
    else:
        densitymap_m = np.zeros(densitymap.shape)
        for i in xrange(densitymap.shape[2]):
            densitymap_m[:,:,i] = densitymap[:,:,densitymap.shape[2]-1-i]
        return densitymap_m
    
def ReadDepth(depthPath,mirror = False,scale = 1.0):
    """
    Load the depth channel from matfile.
    """ 
    depthinfo = sio.loadmat(depthPath)
    depthmap = depthinfo['depth']*scale
    depthmap = depthmap.reshape(1,depthmap.shape[0],depthmap.shape[1])
    if not mirror:
        return depthmap
    else:
        depthmap_m = np.zeros(depthmap.shape)
        for i in xrange(depthmap.shape[2]): 
            depthmap_m[:,:,i] = depthmap[:,:,depthmap.shape[2]-1-i]
        return depthmap_m


def CropSubImage(imArr,dmapArr,downscale = 4.0):
    """
    imArr: channel(rgb = 3,gray = 1) * height * width
    dmapArr: corrsponding downscaled density map
    return  9 sub-images and sub-densitymap
    """
    imShape = imArr.shape
    dmapHeight = dmapArr.shape[1]
    dmapWidth = dmapArr.shape[2]
    imHeight = imShape[1]
    imWidth = imShape[2]
    subimHeight = imHeight/2
    subimWidth = imWidth/2
    subdmapHeight = int(math.ceil(subimHeight/downscale))
    subdmapWidth = int(math.ceil(subimWidth/downscale))
    
    #print subimHeight,subimWidth
    #print dmapHeight,dmapWidth
    #print subdmapHeight,subdmapWidth
    #time.sleep(100)
    
    subimArr1 = imArr[:,0:subimHeight,0:subimWidth]
    subimArr2 = imArr[:,0:subimHeight,imWidth-subimWidth:imWidth]
    subimArr3 = imArr[:,imHeight-subimHeight:imHeight,0:subimWidth]
    subimArr4 = imArr[:,imHeight-subimHeight:imHeight,\
                      imWidth-subimWidth:imWidth]
    subimArr5 = imArr[:,subimHeight/2:subimHeight+subimHeight/2,\
                      0:subimWidth]
    subimArr6 = imArr[:,0:subimHeight,subimWidth/2:\
                      subimWidth+subimWidth/2]
    subimArr7 = imArr[:,imHeight-subimHeight:imHeight,\
                      subimWidth/2:subimWidth+subimWidth/2]
    subimArr8 = imArr[:,subimHeight/2:subimHeight+subimHeight/2,\
                      imWidth-subimWidth:imWidth]
    subimArr9 = imArr[:,subimHeight/2:subimHeight+subimHeight/2,\
                      subimWidth/2:subimWidth+subimWidth/2]
                      
    subdmapArr1 = dmapArr[:,0:subdmapHeight,0:subdmapWidth]
    subdmapArr2 = dmapArr[:,0:subdmapHeight,dmapWidth-subdmapWidth:dmapWidth]
    subdmapArr3 = dmapArr[:,dmapHeight-subdmapHeight:dmapHeight,0:subdmapWidth]
    subdmapArr4 = dmapArr[:,dmapHeight-subdmapHeight:dmapHeight,\
                      dmapWidth-subdmapWidth:dmapWidth]
    subdmapArr5 = dmapArr[:,subdmapHeight/2:subdmapHeight+subdmapHeight/2,\
                      0:subdmapWidth]
    subdmapArr6 = dmapArr[:,0:subdmapHeight,subdmapWidth/2:\
                      subdmapWidth+subdmapWidth/2]
    subdmapArr7 = dmapArr[:,dmapHeight-subdmapHeight:dmapHeight,\
                      subdmapWidth/2:subdmapWidth+subdmapWidth/2]
    subdmapArr8 = dmapArr[:,subdmapHeight/2:subdmapHeight+subdmapHeight/2,\
                      dmapWidth-subdmapWidth:dmapWidth]
    subdmapArr9 = dmapArr[:,subdmapHeight/2:subdmapHeight+subdmapHeight/2,\
                      subdmapWidth/2:subdmapWidth+subdmapWidth/2]
    return subimArr1,subimArr2,subimArr3,subimArr4,subimArr5,subimArr6,\
           subimArr7,subimArr8,subimArr9,subdmapArr1,subdmapArr2,subdmapArr3,\
           subdmapArr4,subdmapArr5,subdmapArr6,subdmapArr7,subdmapArr8,subdmapArr9
           
def CropSubDepth(depthArr):
    """
    depthArr: depth channel for images ( height * width)
    return  9 sub-depthmap
    """
    dmapHeight = depthArr.shape[1]
    dmapWidth = depthArr.shape[2]
    subdmapHeight = dmapHeight/2
    subdmapWidth = dmapWidth/2
                      
    depthArr1 = depthArr[:,0:subdmapHeight,0:subdmapWidth]
    depthArr2 = depthArr[:,0:subdmapHeight,dmapWidth-subdmapWidth:dmapWidth]
    depthArr3 = depthArr[:,dmapHeight-subdmapHeight:dmapHeight,0:subdmapWidth]
    depthArr4 = depthArr[:,dmapHeight-subdmapHeight:dmapHeight,\
                      dmapWidth-subdmapWidth:dmapWidth]
    depthArr5 = depthArr[:,subdmapHeight/2:subdmapHeight+subdmapHeight/2,\
                      0:subdmapWidth]
    depthArr6 = depthArr[:,0:subdmapHeight,subdmapWidth/2:\
                      subdmapWidth+subdmapWidth/2]
    depthArr7 = depthArr[:,dmapHeight-subdmapHeight:dmapHeight,\
                      subdmapWidth/2:subdmapWidth+subdmapWidth/2]
    depthArr8 = depthArr[:,subdmapHeight/2:subdmapHeight+subdmapHeight/2,\
                      dmapWidth-subdmapWidth:dmapWidth]
    depthArr9 = depthArr[:,subdmapHeight/2:subdmapHeight+subdmapHeight/2,\
                      subdmapWidth/2:subdmapWidth+subdmapWidth/2]
    return depthArr1,depthArr2,depthArr3,depthArr4,depthArr5,depthArr6,\
           depthArr7,depthArr8,depthArr9


     
