import cv2
import os
import random
import numpy as np
# from matplotlib import pyplot as plt

from util import sift_get_fmat, err_pts_correspond
# import nsml
# from nsml import DATASET_PATH
from parser import KittiParamParser

def get_FMat(img1, img2, r=1.0, cam1=0, cam2=1, mode='CAM'):
    img1 = cv2.imread(img1,0)  #queryimage # left image
    img1 = cv2.resize(img1, (1392,512), interpolation=cv2.INTER_CUBIC)
    img2 = cv2.imread(img2,0)  #trainimage # right image
    img2 = cv2.resize(img2, (1392,512), interpolation=cv2.INTER_CUBIC)
    
    # p = KittiParamParser(os.path.join(DATASET_PATH,'train','kitti/calib_cam_to_cam.txt'))
    p = KittiParamParser('../data_kitti/kitti/calib_cam_to_cam.txt')
    img1 = p.undistort(img1, cam1)
    img2 = p.undistort(img2, cam2)

    if mode == 'LEMED':
        F, pts1, pts2 = sift_get_fmat(img1, img2, total = 200, ratio = r, algo=cv2.FM_LMEDS)
        # print ("F-matrix (from sift LEMED): \n%s" % F)
        # if F is not None:
        #     err = err_pts_correspond(F, pts1, pts2)
            # print ("pFp avg sum err: %s " %  err)
        # all_pts1, all_pts2 = pts1, pts2

    elif mode == 'RANSAC':
        F, pts1, pts2 = sift_get_fmat(img1, img2, total = 200, ratio = r, algo=cv2.FM_RANSAC)
        # print ("F-matrix (from sift RANSAC): \n%s" % F)
        # if F is not None:
        #     err = err_pts_correspond(F, pts1, pts2)
        #     print ("pFp avg sum err: %s " %  err)
        # all_pts1 = np.concatenate((all_pts1, pts1))
        # all_pts2 = np.concatenate((all_pts2, pts2))

    elif mode == '8POINTS':
        F, pts1, pts2 = sift_get_fmat(img1, img2, total = 200, ratio = r, algo=cv2.FM_8POINT)
        # print ("F-matrix (from sift 8POINTS): \n%s" % F)
        # if F is not None:
        #     err = err_pts_correspond(F, pts1, pts2)
        #     print ("pFp avg sum err: %s " %  err)
        # all_pts1 = np.concatenate((all_pts1, pts1))
        # all_pts2 = np.concatenate((all_pts2, pts2))

    elif mode == '7POINTS':
        F, pts1, pts2 = sift_get_fmat(img1, img2, total = 200, ratio = r, algo=cv2.FM_7POINT)
        # print ("F-matrix (from sift 7POINTS): \n%s" % F)
        # if F is not None:
        #     err = err_pts_correspond(F, pts1, pts2)
        #     print ("pFp avg sum err: %s " %  err)
        # all_pts1 = np.concatenate((all_pts1, pts1))
        # all_pts2 = np.concatenate((all_pts2, pts2))
    
    elif mode == 'CAM':
        F, pts1, pts2 = sift_get_fmat(img1, img2, total = 200, ratio = r, algo=cv2.FM_RANSAC)
        F = p.get_F(cam1, cam2)
        
        # F = np.array([
        #     [1.7e-08, -5.9e-06, -1.59e-03],
        #     [2.54e-06, -8.76e-07, -1.09e-01],
        #     [3.56e-04, 1.12e-01, 1.0]
        # ])
        
        # print ("F-matrix (from camera param):\n %s" % F)
        # print "pFp avg sum err: %s " % err_pts_correspond(F, pts1, pts2)
    
    return F, pts1, pts2
    # demo_epi(img1, img2, F, pts1, pts2)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Basic utility for demonstration purpose')
    parser.add_argument('--img1', type=str, default='left.jpg',  help="The first image")
    parser.add_argument('--img2', type=str, default='right.jpg', help="The second image")
    parser.add_argument('--r',    type=float, default=0.5, help="Ratio for the ratio test.")
    parser.add_argument('--cam1', type=int, default=0, help="ID for first camera")
    parser.add_argument('--cam2', type=int, default=1, help="ID for second camera")
    args = parser.parse_args()

    F, pts1, pts2 = get_FMat(args.img1, args.img2, args.r, args.cam1, args.cam2)
