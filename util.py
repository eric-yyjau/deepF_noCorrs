#!Codes adapted form : http://docs.opencv.org/trunk/da/de9/tutorial_py_epipolar_geometry.html
import os
import cv2
import random
import numpy as np
from povray.stereo_pair import gen_fmat, err_pts_correspond, \
                               find_key_point_pairs, make_cali_img_pairs
from kittiFMat.parser import KittiParamParser
# from matplotlib import pyplot as plt

# CFD = os.path.dirname(os.path.realpath(__file__))
sift = cv2.xfeatures2d.SIFT_create()

# adapted from https://gist.github.com/CannedYerins/11be0c50c4f78cad9549#file-draw_matches-py
def draw_matches(img1, kp1, img2, kp2, color=None):
    """Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.

    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.

    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2

    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 10
    thickness = 1
    if color:
        c = color
    count = 0
    for p1,p2 in zip(kp1, kp2):
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color:
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(p1).astype(int))
        end2 = tuple(np.round(p2).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)

        #   plt.figure(figsize=(15,15))
        #   plt.imshow(new_img)
        #   cv2.imwrite("kp_match_%s.jpg" % count, new_img)
        #   count += 1
        #   plt.show()

    cv2.imwrite("kp_match.jpg", new_img)
    # plt.figure(figsize=(15,15))
    # plt.imshow(new_img)
    # plt.show()

def sift_get_fmat(img1, img2, total=100, ratio = 0.8, algo=cv2.FM_LMEDS,
                  random = False, display = False):
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params  = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < ratio * n.distance:
            good.append(m)

    sorted_good_mat = sorted(good, key=lambda m: m.distance)
    for m in sorted_good_mat:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    # print ('pts size: ', pts1.size)
    assert pts1.size > 2 and pts2.size > 2
    F, mask = cv2.findFundamentalMat(pts1,pts2,algo)
    if mask is None or np.linalg.matrix_rank(F) != 2:
        return None, None, None
    # assert np.linalg.matrix_rank(F) == 2

    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    if random:
        # Randomly sample top-[total] number of points
        pts = random.sample(zip(pts1, pts2), min(len(pts1), total))
        pts1, pts2 = np.array([ p for p, _ in pts ]), \
                     np.array([ p for _, p in pts ])
    else:
        pts1 = pts1[:min(len(pts1), total)]
        pts2 = pts2[:min(len(pts1), total)]

    if display:
        draw_matches(img1,pts1,img2,pts2)

    return F, pts1, pts2

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        cv2.line(img1, (x0,y0), (x1,y1), color,1)
        cv2.circle(img1,tuple(pt1),5,color,-1)
        cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def demo_epi(img1, img2, F, pts1, pts2):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

    #plt.figure()
    # plt.subplot(121)
    # plt.imshow(img5)

    #plt.figure()
    # plt.subplot(122)
    # plt.imshow(img3)

    # plt.show()
'''
if __name__ == "__main__":
    img1 = cv2.imread('data/kitti/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000030.png',0)  #queryimage # left image
    img2 = cv2.imread('data/kitti/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/0000000030.png',0)  #queryimage # left image

    # img1 = cv2.imread('im0.png',0)  #queryimage # left image
    # img2 = cv2.imread('im1.png',0)  #trainimage # right image

    # img1 = cv2.imread('left.jpg',0)  #queryimage # left image
    # img2 = cv2.imread('right.jpg',0)  #trainimage # right image

    #   filename = "pomme"
    #   img1 = cv2.imread(os.path.join(CFD, 'povray', filename, 'left.png'),0)  #queryimage # left image
    #   img2 = cv2.imread(os.path.join(CFD, 'povray', filename, 'right.png'),0)  #trainimage # right image

    r = 0.5
    total = 10

    print "Computing using LEMED..."
    F1, pts1, pts2 = sift_get_fmat(img1, img2, total = total, ratio = r, algo=cv2.FM_LMEDS)
    all_pts1, all_pts2 = pts1 , pts2

    print "Computing using RANSAC..."
    F2, pts1, pts2 = sift_get_fmat(img1, img2, total = total, ratio = r, algo=cv2.FM_RANSAC)
    all_pts1 = np.concatenate((all_pts1, pts1))
    all_pts2 = np.concatenate((all_pts2, pts2))

    print "Computing using 8-points algorithm"
    F3, pts1, pts2 = sift_get_fmat(img1, img2, total = total, ratio = r, algo=cv2.FM_8POINT)
    all_pts1 = np.concatenate((all_pts1, pts1))
    all_pts2 = np.concatenate((all_pts2, pts2))

    print "Computing using 7-points algorithm"
    F4, pts1, pts2 = sift_get_fmat(img1, img2, total = total, ratio = r, algo=cv2.FM_7POINT)
    all_pts1 = np.concatenate((all_pts1, pts1))
    all_pts2 = np.concatenate((all_pts2, pts2))

    print "Computing using camera parameters"
    p = KittiParamParser("data/kitti/2011_09_26/calib_cam_to_cam.txt")
    F5 = p.get_F(2, 3)

    #   w, h    = 512, 512
    #   loc     = [0, 10, -30]
    #   look_at = [0, 4, 0]
    #   dir1    = [0, 0, 1]
    #   dir2    = [0, 0, 1]
    #   trans   = [1, 0, 1]
    #   rotate  = [1, 1, 0]
    #   F5      = gen_fmat(w, h, loc, look_at, dir1, dir2, rotate = rotate, trans = trans)

    # all_pts1, all_pts2 = find_key_point_pairs(10, w, h, loc, look_at,
    #                    dir1, dir2, rotate = rotate, trans = trans, rand_amount = 3)

    draw_matches(img1, all_pts1, img2, all_pts2)
    print "F-matrix (from sift LEMED): \n%s" % F1
    print "pFp avg sum err: %s " % err_pts_correspond(F1, all_pts1, all_pts2)
    print "F-matrix (from sift RANSAC): \n%s" % F2
    print "pFp avg sum err: %s " % err_pts_correspond(F2, all_pts1, all_pts2)
    print "F-matrix (from sift 8POINTS): \n%s" % F3
    print "pFp avg sum err: %s " % err_pts_correspond(F3, all_pts1, all_pts2)
    print "F-matrix (from sift 7POINTS): \n%s" % F4
    print "pFp avg sum err: %s " % err_pts_correspond(F4, all_pts1, all_pts2)
    print "F-matrix (from camera param):\n %s" % F5
    print "pFp avg sum err: %s " % err_pts_correspond(F5, all_pts1, all_pts2)
'''
