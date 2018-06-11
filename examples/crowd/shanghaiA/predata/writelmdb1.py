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
def WriteLmdbv1(imPath,dmapPath,lmdbPth,dataset,downscale,mirror):
    """
    imPath:   the folder contains images.
    dmapPath: the folder contains matfiles of density map. 
    lmdbPth:  the folder will be used to save lmdb file.
    num: the count of images.
    mirror: contains mirror data
    """				
    print('Start writing lmdb') 
    patches=[]
    dmaps=[]
    imLmdbName = lmdbPth + '/image_lmdb'
    dmapLmdbName = lmdbPth + '/dmap_lmdb'
    scale=1
    for idx in dataset:
	    imName = imPath+str(idx)+'.jpg'
	    dmapName = dmapPath + str(idx)+'.mat'
	    imArr = impro.ReadImage(imName,False,scale)
	    dmapArr = impro.ReadDmap(dmapName,False,scale)
	    data = impro.CropSubImage(imArr,dmapArr,downscale)
	    if mirror:
			    imArr_m = impro.ReadImage(imName,True,scale)
			    dmapArr_m = impro.ReadDmap(dmapName,True,scale)
			    data_m = impro.CropSubImage(imArr_m,dmapArr_m,downscale)
	    for in_idx in xrange(9):
			    patches.append(data[in_idx])
			    dmaps.append(data[in_idx+9])
			    if mirror:
			    	    patches.append(data_m[in_idx])		    
			    	    dmaps.append(data_m[in_idx+9])
	    string_ = str(idx)
	    sys.stdout.write("\r%s" % string_)
	    sys.stdout.flush()
    print(len(patches))
    print('\n Saving! \n')
    r = np.random.permutation(len(patches))
    imglmdb = lmdb.open(imLmdbName,map_size = int(1e12))
    with imglmdb.begin(write=True) as in_txn:  
		      for in_idx in xrange(len(patches)):
				    datum = caffe.io.array_to_datum(patches[r[in_idx]])
				    str_id = '{:0>10d}'.format(in_idx)
			  	    in_txn.put(str_id,datum.SerializeToString())		      
    imglmdb.close()
    dmaplmdb = lmdb.open(dmapLmdbName,map_size = int(1e12))
    with dmaplmdb.begin(write=True) as in_txn:
		        for in_idx in xrange(len(patches)):
				    datum = caffe.io.array_to_datum(dmaps[r[in_idx]]) 
				    str_id = '{:0>10d}'.format(in_idx)
				    in_txn.put(str_id,datum.SerializeToString())		        
    dmaplmdb.close()
    """f = h5py.File(lmdbPth, 'w')
    f.create_dataset('data', data=patches, dtype=np.float32)
    f.create_dataset('label',data=dmaps, dtype=np.float32)"""
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
