import argparse
import sys
sys.path.insert(0,'../../../test_caffe_python3.6/caffe/python')
import caffe
import scipy.misc
import numpy as np
import time

parser = argparse.ArgumentParser()

parser.add_argument("--gpu", type=int)

args = parser.parse_args()

caffe.set_mode_gpu()
caffe.set_device(args.gpu)

proto = '../../../test_cuda_ucn/for_prakash_models/googlenet_ucn.prototxt'
wt = '../../../test_cuda_ucn/for_prakash_models/googlenet_ucn_enhanced.caffemodel'
caffe_net = caffe.Net(proto, wt, caffe.TEST)

downsampling_factor = 4.0
PIXEL_MEANS = np.array([[[94.95408014, 99.54709961, 94.48018654]]])

def transform_coord_im2feat(self, x):
    # print ('in hdp trans')
    """ image coord to feature coord"""
    # return (x + 0.5) / self._downsampling_factor - 0.5
    return x / downsampling_factor

def transform_coord_feat2im(self, x):
    """ transform feature coord to image coord """
    # print ('in hdp feat2im')
    return x * downsampling_factor

def ind2coord(self, ind, width):
    """ takse in num_batch x num_coords x k ind and genertes num_batch x
    num_coords x 2"""
    # print ('in hdp ind2cord')
    num_batch, num_coord, k = ind.shape

    # print (num_batch, num_coord, ind.shape)
    # print ('after ind ', ind)
    # k-th NN index: ind[:, k, :]
    x = ind % width
    y = np.floor(ind / width)
    # print (ind[0,0,0], x[0,0,0], y[0,0,0], width)
    xy_coords = np.concatenate((x[..., np.newaxis], y[..., np.newaxis]), axis=3)
    # print ('xy ',xy_coords.shape, xy_coords)
    return xy_coords

img1 = './image1'

image_1 = scipy.misc.imread(img1).astype('float32')
image_1 = scipy.misc.imresize(image_1, (224, 224)) - PIXEL_MEANS
image_2 = scipy.misc.imread(img2).astype('float32')
image_2 = scipy.misc.imresize(image_2, (224, 224)) - PIXEL_MEANS

image_1 = np.expand_dims(image_1, axis=0)
image_2 = np.expand_dims(image_2, axis=0)
image_1 = np.array(image_1).transpose(0,3,1,2)
image_2 = np.array(image_2).transpose(0,3,1,2)

caffe_net.blobs['image_1'].data[...] = image_1
caffe_net.blobs['image_2'].data[...] = image_2

caffe_net.forward()

knn_inds = caffe_net.blobs['ind'].data
feature1 = caffe_net.blobs['feature1'].data
feature2 = caffe_net.blobs['feature2'].data

knn_inds_shape = knn_inds.shape
knn = knn_inds_shape[1]
_, _, height, width = feature1.shape

# Conver to feature xy_coord
# print ('before ', knn_inds)
feat2_xy_coord_knn = ind2coord(knn_inds - 1, width)

# print (knn_inds.shape, knn_inds)
# print (feat2_xy_coord_knn.shape, feat2_xy_coord_knn)

# Conver to the image coord
img2_xy_coord_knn = transform_coord_feat2im(feat2_xy_coord_knn)
img2_xy_coord_gt = correspondences[:, :, 2:4]
feat2_xy_coord_gt = transform_coord_im2feat(img2_xy_coord_gt)

