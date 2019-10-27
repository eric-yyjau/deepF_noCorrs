import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

from util import sift_get_fmat, err_pts_correspond
from parser import KittiParamParser

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Basic utility for demonstration purpose')
    parser.add_argument('--img1', type=str, default='left.jpg',  help="The first image")
    parser.add_argument('--img2', type=str, default='right.jpg', help="The second image")
    parser.add_argument('--r',    type=float, default=0.5, help="Ratio for the ratio test.")
    parser.add_argument('--cam1', type=int, default=0, help="ID for first camera")
    parser.add_argument('--cam2', type=int, default=1, help="ID for second camera")
    args = parser.parse_args()

    img1 = cv2.imread(args.img1,0)  #queryimage # left image
    img2 = cv2.imread(args.img2,0)  #trainimage # right image
    r = args.r

    F, pts1, pts2 = sift_get_fmat(img1, img2, total = 200, ratio = r)
    print ("F-matrix (from sift LEMED): \n%s" % F)
    err = err_pts_correspond(F, pts1, pts2)
    print ("pFp avg sum err: %s " %  err)
    all_pts1, all_pts2 = pts1, pts2

    F, pts1, pts2 = sift_get_fmat(img1, img2, total = 200, ratio = r, algo=cv2.FM_RANSAC)
    print ("F-matrix (from sift RANSAC): \n%s" % F)
    err = err_pts_correspond(F, pts1, pts2)
    print ("pFp avg sum err: %s " %  err)
    all_pts1 = np.concatenate((all_pts1, pts1))
    all_pts2 = np.concatenate((all_pts2, pts2))

    F, pts1, pts2 = sift_get_fmat(img1, img2, total = 200, ratio = r, algo=cv2.FM_8POINT)
    print ("F-matrix (from sift 8POINTS): \n%s" % F)
    err = err_pts_correspond(F, pts1, pts2)
    print ("pFp avg sum err: %s " %  err)
    all_pts1 = np.concatenate((all_pts1, pts1))
    all_pts2 = np.concatenate((all_pts2, pts2))

    F, pts1, pts2 = sift_get_fmat(img1, img2, total = 200, ratio = r, algo=cv2.FM_7POINT)
    print ("F-matrix (from sift 7POINTS): \n%s" % F)
    err = err_pts_correspond(F, pts1, pts2)
    print ("pFp avg sum err: %s " %  err)
    all_pts1 = np.concatenate((all_pts1, pts1))
    all_pts2 = np.concatenate((all_pts2, pts2))

    p = KittiParamParser("../data/kitti/2011_09_26/calib_cam_to_cam.txt")
    F = p.get_F(args.cam1, args.cam2)
    F = np.array([
        [1.7e-08, -5.9e-06, -1.59e-03],
        [2.54e-06, -8.76e-07, -1.09e-01],
        [3.56e-04, 1.12e-01, 1.0]
    ])
    print ("F-matrix (from camera param):\n %s" % F)
    print ("pFp avg sum err: %s " % err_pts_correspond(F, all_pts1, all_pts2))

    # demo_epi(img1, img2, F, pts1, pts2)
