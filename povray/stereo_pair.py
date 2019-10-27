"""
This file contains the utility to generate stero pairs from povray

Note: [0,0,0] is Black, [255, 255, 255] is white.

"""
import os
import cv2
import itertools
import numpy as np
import vapory as vp
# from matplotlib import pyplot as plt
CFD = os.path.dirname(os.path.realpath(__file__))

def get_rotation_mat(rx, ry, rz):
    """
    [param]
        [rx]    (float) rotation of x-axis in degree
        [ry]    (float) rotation of y-axis in degree
        [rz]    (float) rotation of z-axis in degree
    [return]
        3x3 np.array (float32)  3-D rotation matrix
    [reference]
        https://en.wikipedia.org/wiki/Rotation_matrix
    """
    def _to_radius(x):
        return x / 180. * np.pi
    rx, ry, rz = _to_radius(rx), _to_radius(ry), _to_radius(rz)

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    R_y = np.array([
        [np.cos(ry),  0, -np.sin(ry)],
        [0, 1, 0],
        [np.sin(ry), 0, np.cos(ry)]
    ])

    #   R_y = np.array([
    #       [np.cos(ry),  0, -np.sin(ry)],
    #       [0, 1, 0],
    #       [np.sin(ry), 0, np.cos(ry)]
    #   ])

    R_z = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz),  0],
        [0, 0, 1]
    ])
    R = np.dot(np.dot(R_x, R_y), R_z)

    # assert False
    return R

def get_translate_mat(tx, ty, tz):
    """
    [param]
        [tx]    (float) translate in x-axis
        [ty]    (float) translate in y-axis
        [tz]    (float) translate in z-axis
    [return]
        3x3 np.array (float32) 3-D translation matrix
    """
    return np.array([
        [0,   -tz,  ty],
        [tz,  0,    -tx],
        [-ty, tx,   0]
    ]).astype(np.float64)

def get_intrinsic_mat(povray_cam):
    """
    Take parameters that specified a pair
    """
    direction = povray_cam['direction']
    f = np.linalg.norm(direction)
    c_x, c_y = 0.   # using the same coordinate system
    alpha = 0.      # square pixel
    # skew  = 1.0     # normal pixel size
    return np.array([
        [-f, 0,         c_x],
        [0, -alpha * f, c_y],
        [0, 0, 1]
    ])

def get_rel_rotate(vec1, vec2):
    """
    [precondition]  unit(vec1) + unit(vec2) != 0
    Reference: http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    [postcondition] return R is the rotation matrix that rotate vec1 to vec2
    """
    uv1 = vec1 / np.linalg.norm(vec1)
    uv2 = vec2 / np.linalg.norm(vec2)
    assert not(np.allclose(uv1 + uv2, np.zeros(uv1.shape)))
    v   = np.cross(uv1, uv2)
    # s   = np.linalg.norm(v)
    c   = np.dot(uv1, uv2)
    Vx  = np.array([
        [0,     -v[2],  v[1]],
        [v[2],  0,      -v[0]],
        [-v[1], v[0],   0]
    ]).astype(np.float64)
    R   = np.identity(3) + Vx + np.dot(Vx, Vx) / (1 + c)
    assert np.allclose(np.dot(R, uv1), uv2)
    return R

def gen_fmat(w, h, loc, look_at, direct1, direct2,
             rotate = [0, 0, 0], trans = [0,0,0], ctype = None):
    """
    Calculate fundamental matrices given the parameters needed to compute Fmat

    Assuming the template follows the following format:
        location:   <location vector>
        direction:  <init direction vector>
        translate:  <translate vector>
        rotate:     <rotate vector>
        look_at:    <look at vector>

    Camera I :  loc + direct1 + trans + rotate + look_at
    Camera II:  loc + direct2 + trans + rotate + look_at
    These two cameras share the same <look_at>
    """
    look_at = np.array(look_at)
    loc     = np.array(loc)

    loc1 = np.array(loc)
    f_dir1 = look_at - loc

    Rfor   = get_rotation_mat(rotate[0], rotate[1], rotate[2])
    loc2   = np.dot(Rfor, loc)
    f_dir2 = look_at - loc2
    print("LOC2:\n%s %s" % (loc2, np.dot(Rfor, loc)))

    R = get_rel_rotate(f_dir2, f_dir1)
    print("R:\n%s" % R)

    T = loc2 + trans - loc1
    print("Translate : %s " % T)
    Tx= get_translate_mat(T[0], T[1], T[2])
    print("Tx:\n%s" % Tx)

    f1 = np.linalg.norm(direct1)
    if ctype == "CCD":
        K1= np.array([
            [-f1*w,     0,      w / 2.],
            [0,         -f1*h,  h / 2.],
            [0,         0,      1]
        ])
    else:
        K1= np.array([
            [-f1,       0,      w / 2.],
            [0,         -f1,    h / 2.],
            [0,         0,      1]
        ])
    print("K1:\n%s" % K1)
    K1inv = np.linalg.inv(K1)

    f2 = np.linalg.norm(direct2)
    if ctype == "CCD":
        K2= np.array([
            [-f2*w,     0,      w / 2.],
            [0,         -f2*h,  h / 2.],
            [0,         0,      1]
        ])
    else:
        K2= np.array([
            [-f2,       0,      w / 2.],
            [0,         -f2,    h / 2.],
            [0,         0,      1]
        ])
    print("K2:\n%s" % K2)
    K2inv = np.linalg.inv(K2)

    F  = np.dot(K2inv.T, np.dot(R, np.dot(Tx, K1inv)))
    print("F-matrix (not normalized) :\n%s" % F)
    F /= F[2, 2]
    assert np.linalg.matrix_rank(F) == 2
    return F

def get_kp_cali_img(img, verbose = 0):
    """
    Calibration images are images with only one point that's not background
    """
    w, h = img.shape[:2]
    img  = np.ones(img.shape) * 255 - img
    ret = [[i,j] for i,j in itertools.product(range(w), range(h)) if np.sum(img[i,j]) > 0]
    ret = np.array(ret)
    mean_pt = np.mean(ret, axis = 0)
    if verbose > 0:
        print(len(ret), mean_pt)
    # print(np.var(ret, axis=0))
    return mean_pt, len(ret)

def make_cali_img_pairs(w, h, loc, look_at, direct1, direct2,
                        rotate = [0, 0, 0], trans = [0,0,0], ctype = None, rand_amount = 1):
    cali_dir = os.path.join(CFD, "caliimg")
    light  = vp.LightSource(
        # [2,4,-3],
        [0, 0, -10000],
        'color', "White", 'rotate', [30,0,0], 'rotate', [0,88,0] )      # White light

    background = vp.Background("color", "White") # White background

    center = np.array(look_at) + (np.random.rand(3) - 0.5) * rand_amount
    sphere = vp.Sphere(
        center, 0.1,                                    # center, radius
        vp.Texture( vp.Pigment( 'color', "Black" )))    # Black point

    l_camera = vp.Camera(
        'location',   loc,
        'direction',  direct1,
        'up',         [0,1,0],
        'right',      [1*w/h, 0, 0],
        'look_at',    look_at)
    l_scene  = vp.Scene(l_camera, objects= [background, light, sphere],
                        included = ["colors.inc"])
    l_img_path = os.path.join(cali_dir, "left.png")
    l_scene.render(l_img_path, width=w, height=h, auto_camera_angle=False)

    r_camera = vp.Camera(
        'location',   loc,
        'direction',  direct2,
        'up',         [0,1,0],
        'right',      [1*w/h, 0, 0],
        'look_at',    look_at,
        'rotate',     rotate,
        'translate',  trans
    )
    r_scene  = vp.Scene(r_camera, objects= [background, light, sphere],
                        included = ["colors.inc"])
    r_img_path = os.path.join(cali_dir, "right.png")
    r_scene.render(r_img_path, width=w, height=h, auto_camera_angle=False)

    with open(os.path.join(cali_dir, "left.pov"), "wb") as f:
        f.write(l_scene.__str__())

    with open(os.path.join(cali_dir, "right.pov"), "wb") as f:
        f.write(r_scene.__str__())

    return cv2.imread(l_img_path), cv2.imread(r_img_path)


def abserr_pts_correspond(F, pts1, pts2):
    """
    err_pts_correspond(F, p, q, norm=n) will return the estimated errors
    computed by 1/n sum_i (q_i^T F p_i)^2
    """
    assert len(pts1) == len(pts2)
    err = 0.0
    for p1, p2 in zip(pts1, pts2):
        hp1, hp2 = np.ones(3), np.ones(3)
        hp1[:2], hp2[:2] = p1, p2
        err += np.abs(np.dot(hp2.T, np.dot(F, hp1)))
    return err / float(len(pts1))


def err_pts_correspond(F, pts1, pts2):
    """
    err_pts_correspond(F, p, q, norm=n) will return the estimated errors
    computed by 1/n sum_i (q_i^T F p_i)^2
    """
    assert len(pts1) == len(pts2)
    err = 0.0
    for p1, p2 in zip(pts1, pts2):
        hp1, hp2 = np.ones(3), np.ones(3)
        hp1[:2], hp2[:2] = p1, p2
        err += np.dot(hp2.T, np.dot(F, hp1))**2
        # err += np.abs(np.dot(hp2.T, np.dot(F, hp1)))
    return err / float(len(pts1))

def lin_recon_err(F, pts1, pts2):
    """
    Compute the linear reconstruction error using the key-points.
    """
    assert len(pts1) == len(pts2)
    err = 0.0
    e3  = np.array([0,0,1.])
    for p1, p2 in zip(pts1, pts2):
        hp1, hp2 = np.ones(3), np.ones(3)
        hp1[:2], hp2[:2] = p1, p2
        d = np.dot(hp2.T, np.dot(F, hp1))**2
        err +=  d / np.dot(np.dot(e3.T, F), hp1)**2 + \
                d / np.dot(np.dot(hp2.T, F), e3)**2

    return err / float(len(pts1))


def sym_epipolar_dist(F, pts1, pts2, epsilon=1e-5):
    assert len(pts1) == len(pts2)
    err = 0.
    for p1, p2 in zip(pts1, pts2):
        hp1, hp2 = np.ones(3), np.ones(3)
        hp1[:2], hp2[:2] = p1, p2
        fp, fq = np.dot(F, hp1), np.dot(F.T, hp2)
        sym_jjt = 1./(fp[0]**2 + fp[1]**2 + epsilon) + 1./(fq[0]**2 + fq[1]**2 + epsilon)
        err = err + ((np.dot(hp2.T, np.dot(F, hp1))**2) * (sym_jjt + epsilon))
    '''
    print ('F: ', F)
    print ('fp: ', fp, 'sum: ', fp[0]**2 + fp[1]**2)
    print ('fq: ', fq, 'sum: ', fq[0]**2 + fq[1]**2)
    print ('sym jjt: ', sym_jjt)
    print ('p1: ', hp1)
    print ('p2: ', hp2)
    print ('err: ', np.dot(hp2.T, np.dot(F, hp1)), 'len: ', len(pts1))
    print ('err squared: ', np.dot(hp2.T, np.dot(F, hp1))**2, 'len: ', len(pts1))
    print ('delta err term: ', ((np.dot(hp2.T, np.dot(F, hp1))**2) * (sym_jjt + epsilon)))
    print ('total err: ', err, 'normalised error: ', err / float(len(pts1)))
    '''
    return err / float(len(pts1))

def sampson_dist(F, pts1, pts2, epsilon=1e-5):
    """
    implements the Sampson Distance (first order approximation) evaluation of
    the F-matrices
    """
    assert len(pts1) == len(pts2)
    err = 0.0
    for p1, p2 in zip(pts1, pts2):
        hp1, hp2 = np.ones(3), np.ones(3)
        hp1[:2], hp2[:2] = p1, p2
        fp, fq = np.dot(F, hp1), np.dot(F.T, hp2)
        JJT = fp[0]**2 + fp[1]**2 + fq[0]**2 + fq[1]**2
        err = err + ((np.dot(hp2.T, np.dot(F, hp1))**2) / (JJT + epsilon))
    return err / float(len(pts1))

def find_key_point_pairs(n, w, h, loc, look_at, dir1, dir2,
                        rotate = [0, 0, 0], trans = [0,0,0], ctype = None,
                        display = False, rand_amount = 1):
    kpts1, kpts2 = [], []
    while len(kpts1) < n:
        left, right = make_cali_img_pairs(w, h, loc, look_at,
                      dir1, dir2, rotate = rotate, trans = trans, rand_amount = rand_amount)
        (lp,ll), (rp, rl) = get_kp_cali_img(left), get_kp_cali_img(right)
        if ll == 0 or ll > 10:
            continue
        if rl == 0 or rl > 10:
            continue
        print("%s key point pairs:%s\t%s" % (len(kpts1) + 1, (lp, ll), (rp, rl)))
        kpts1.append(lp)
        kpts2.append(rp)
        '''
        if display:
            plt.figure(figsize=(10,5))
            plt.subplot(121)
            plt.imshow(left)
            plt.title("Left Image:%s from %s" % (lp, ll))
            plt.subplot(122)
            plt.imshow(right)
            plt.title("Right Image:%s from %s" % (rp, rl))
            plt.show()
        '''
    return kpts1, kpts2

if __name__ == "__main__":
    w, h    = 512, 512
    loc     = [0, 10, -30]
    look_at = [0, 4, 0]
    dir1    = [0, 0, 1]
    dir2    = [0, 0, 1]
    trans   = [1, 0, 1]
    rotate  = [1, 1, 0]
    F = gen_fmat(w, h, loc, look_at, dir1, dir2, rotate = rotate, trans = trans)
    kpts1, kpts2 = find_key_point_pairs(10, w, h, loc, look_at, dir1, dir2, rotate = rotate, trans = trans, display=False, rand_amount = 20)
    print("unnormalized err :%s" % err_pts_correspond(F, kpts1, kpts2))


