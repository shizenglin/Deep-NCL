# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:07:23 2015

@author: zhangyingying

Wtrite data into lmdb
"""

import sys
caffe_root = '/home/peiyong/Work/Zenglin/caffe-szl/'
sys.path.insert(0,caffe_root+'python')
import caffe
import lmdb
import numpy as np
import imageprocess as impro
import math
import h5py
from PIL import Image
import scipy.io as sio
from skimage.transform import resize
def WriteLmdbTrain(imPath,dmapPath,lmdbPth,dataset,downscale,mirror):
    """
    imPath:   the folder contains images.
    dmapPath: the folder contains matfiles of density map. 
    lmdbPth:  the folder will be used to save lmdb file.
    num: the count of images.
    mirror: contains mirror data
    """				
    print('Start writing lmdb\n') 
    imLmdbName = lmdbPth + '/image_lmdb'
    dmapLmdbName = lmdbPth + '/dmap_lmdb'
    scales=[1]#[0.6,0.7,0.8,0.9,1.0,1.1,1.2]
    num=len(dataset)*18*len(scales)
    print(num)
    np.random.seed(9999)
    randidx = np.random.permutation(range(1,num+1))
    for idx in xrange(len(dataset)):
	    imName = imPath+str(dataset[idx])+'.jpg'
	    dmapName = dmapPath + str(dataset[idx])+'.mat'
	    for sid in xrange(len(scales)):
		    imArr = impro.ReadImage(imName,False,scales[sid])
	    	    dmapArr = impro.ReadDmap(dmapName,False,scales[sid])
	    	    data = impro.CropSubImage(imArr,dmapArr,downscale)
	    	    imArr_m = impro.ReadImage(imName,True,scales[sid])
	    	    dmapArr_m = impro.ReadDmap(dmapName,True,scales[sid])
	    	    data_m = impro.CropSubImage(imArr_m,dmapArr_m,downscale)
	    	    imglmdb = lmdb.open(imLmdbName,map_size = int(1e12))
	    	    with imglmdb.begin(write=True) as in_txn:  
	    		        for in_idx in xrange(9):
	    				    datum = caffe.io.array_to_datum(data[in_idx])
	    				    str_id = '{:0>8d}'.format(randidx[idx*18*len(scales)+sid*18+in_idx])#[idx*108+sid*18+in_idx]
	    			  	    in_txn.put(str_id,datum.SerializeToString())
	    		        for in_idx in xrange(9):
	    			        datum = caffe.io.array_to_datum(data_m[in_idx])
	    			        str_id = '{:0>8d}'.format(randidx[idx*18*len(scales)+sid*18+in_idx+9])
	    			        in_txn.put(str_id,datum.SerializeToString())
	    	    imglmdb.close()
	    	    dmaplmdb = lmdb.open(dmapLmdbName,map_size = int(1e12))
	    	    with dmaplmdb.begin(write=True) as in_txn:
	    		        for in_idx in xrange(9):
	    				    datum = caffe.io.array_to_datum(data[in_idx+9]) 
	    				    str_id = '{:0>8d}'.format(randidx[idx*18*len(scales)+sid*18+in_idx])
	    				    in_txn.put(str_id,datum.SerializeToString())
	    		        for in_idx in xrange(9):
	    				    datum = caffe.io.array_to_datum(data_m[in_idx+9]) 
	    				    str_id = '{:0>8d}'.format(randidx[idx*18*len(scales)+sid*18+in_idx+9])
	    				    in_txn.put(str_id,datum.SerializeToString())
	    	    dmaplmdb.close()
    	    string_ = str(idx+1)+'/'+str(len(dataset))
    	    sys.stdout.write("\r%s" % string_)
    	    sys.stdout.flush()
    print('\n Finish! \n')
    
def WriteLmdbTest(imPath,dmapPath,lmdbPth,dataset,downscale,mirror):
    """
    imPath:   the folder contains images.
    dmapPath: the folder contains matfiles of density map. 
    lmdbPth:  the folder will be used to save lmdb file.
    num: the count of images.
    mirror: contains mirror data
    """				
    print('Start writing lmdb\n') 
    imLmdbName = lmdbPth + '/image_lmdb'
    dmapLmdbName = lmdbPth + '/dmap_lmdb'
    for idx in xrange(len(dataset)):
	    imName = imPath+str(dataset[idx])+'.jpg'
	    dmapName = dmapPath + str(dataset[idx])+'.mat'
	    imArr = impro.ReadImage(imName)
	    dmapArr = impro.ReadDmap(dmapName)
	    data = impro.CropSubImage(imArr,dmapArr,downscale)
	    imglmdb = lmdb.open(imLmdbName,map_size = int(1e12))
	    with imglmdb.begin(write=True) as in_txn:  
    		        for in_idx in xrange(9):
    				    datum = caffe.io.array_to_datum(data[in_idx])
    				    str_id = '{:0>6d}'.format(idx*9+in_idx)
    				    in_txn.put(str_id,datum.SerializeToString())
	    imglmdb.close()
	    dmaplmdb = lmdb.open(dmapLmdbName,map_size = int(1e12))
	    with dmaplmdb.begin(write=True) as in_txn:
    		        for in_idx in xrange(9):
    				    datum = caffe.io.array_to_datum(data[in_idx+9]) 
    				    str_id = '{:0>6d}'.format(idx*9+in_idx)
    				    in_txn.put(str_id,datum.SerializeToString())		      
	    dmaplmdb.close()
	    string_ = str(idx+1)+'/'+str(len(dataset))
	    sys.stdout.write("\r%s" % string_)
	    sys.stdout.flush()
    print('\n Finish! \n')

def WriteLmdbTestV2(imPath,dmapPath,lmdbPth,dataset,downscale,mirror):
    """
    imPath:   the folder contains images.
    dmapPath: the folder contains matfiles of density map. 
    lmdbPth:  the folder will be used to save lmdb file.
    num: the count of images.
    mirror: contains mirror data
    """				
    print('Start writing lmdb\n') 
    imLmdbName = lmdbPth + '/image_lmdb'
    dmapLmdbName = lmdbPth + '/dmap_lmdb'
    for idx in xrange(len(dataset)):
	    imName = imPath+str(dataset[idx])+'.jpg'
	    dmapName = dmapPath + str(dataset[idx])+'.mat'
	    imArr = impro.ReadImage(imName)
	    dmapArr = impro.ReadDmap(dmapName)
	    imglmdb = lmdb.open(imLmdbName,map_size = int(1e12))
	    with imglmdb.begin(write=True) as in_txn:  
    		        datum = caffe.io.array_to_datum(imArr)
    		        str_id = '{:0>6d}'.format(idx)
    		        in_txn.put(str_id,datum.SerializeToString())
	    imglmdb.close()
	    dmaplmdb = lmdb.open(dmapLmdbName,map_size = int(1e12))
	    with dmaplmdb.begin(write=True) as in_txn:
    		        datum = caffe.io.array_to_datum(dmapArr) 
    		        str_id = '{:0>6d}'.format(idx)
    		        in_txn.put(str_id,datum.SerializeToString())		      
	    dmaplmdb.close()
	    string_ = str(idx+1)+'/'+str(len(dataset))
	    sys.stdout.write("\r%s" % string_)
	    sys.stdout.flush()
    print('\n Finish! \n')


def WriteLmdbv2(imPath,dmapPath,lmdbPth,num,downscale):
    """
    imPath:   the folder contains images.
    dmapPath: the folder contains matfiles of density map. 
    lmdbPth:  the folder will be used to save lmdb file.
    num: the count of images.
    mirror: contains mirror data
    """				
    print('Start writing lmdb') 
    randidx = np.random.permutation(range(1,num+1))
    randidx_m = np.random.permutation(range(1,num+1))
    for idx in xrange(num):
	    imName = imPath + '/IMG_' + str(randidx[idx])+'.jpg'
	    dmapName = dmapPath + '/DMAP_' + str(randidx[idx])+'.mat'
	    imLmdbName = lmdbPth + '/image_lmdb'
	    dmapLmdbName = lmdbPth + '/dmap_lmdb'
	    imArr = impro.ReadImage(imName,'Gray')
	    dmapArr = impro.ReadDmap(dmapName)
	    imArr_m = impro.ReadImage(imName,'Gray',True)
	    dmapArr_m = impro.ReadDmap(dmapName,True)
	    imglmdb = lmdb.open(imLmdbName,map_size = int(1e12))
	    with imglmdb.begin(write=True) as in_txn:  
		    datum = caffe.io.array_to_datum(imArr)
		    str_id = '{:0>10d}'.format(idx*2)
	  	    in_txn.put(str_id,datum.SerializeToString())
	  	    datum = caffe.io.array_to_datum(imArr_m)
		    str_id = '{:0>10d}'.format(idx*2+1)
		    in_txn.put(str_id,datum.SerializeToString())
	    imglmdb.close()

	    dmaplmdb = lmdb.open(dmapLmdbName,map_size = int(1e12))
	    with dmaplmdb.begin(write=True) as in_txn:
		    datum = caffe.io.array_to_datum(dmapArr)
		    str_id = '{:0>10d}'.format(idx*2)
	  	    in_txn.put(str_id,datum.SerializeToString())
	  	    datum = caffe.io.array_to_datum(dmapArr_m)
		    str_id = '{:0>10d}'.format(idx*2+1)
		    in_txn.put(str_id,datum.SerializeToString())
	    dmaplmdb.close()
	    string_ = str(idx+1)+'/'+str(num)
	    sys.stdout.write("\r%s" % string_)
	    sys.stdout.flush()
    print('\n Finish! \n')
