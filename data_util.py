"""
This file contains the codes for preprocessing.
"""
import sys
import os
import cv2
import time
import json
# import cPickle
import pickle
# import pykitti
import numpy as np
import progressbar
# import nsml
# from nsml import DATASET_PATH

from kittiFMat.kitti_fmat import get_FMat

# CFD = os.path.dirname(os.path.realpath(__file__))

# PATH_CAM = DATASET_PATH+'/train/kitti/2011_09_26/'
# PATH_CAM = '../data_kitti/kitti/2011_09_26/'
# PATH_CAM = './data/kitti/2011_09_26_drive_0001_sync/'
PATH_CAM = './data/kitti/2011_09_26/'
# PATH_CAM_LEFT = 'data/kitti/2011_09_26/2011_09_26_drive_0001_sync/image_00/data/'
# PATH_CAM_RIGHT = 'data/kitti/2011_09_26/2011_09_26_drive_0001_sync/image_01/data/'

PATH_ALOI = 'data/grey'

PATH_MVS = 'data/SampleSet/MVS_Data/Cleaned'

'''
def bind_model(feature):

    def save(filename, **kwargs):
        np.save(os.path.join(DATASET_PATH, filename), feature)
        print ('Saved features')

    def load(filename, **kwargs):
        obj = np.load(filename)
        print ('Loaded features')
        return obj

    nsml.bind(save=save, load=load)
'''

def data_spliter(data_lst, tr_ratio = 0.8, shuffle = True):
    """
    Takes all the possible data X, Y, return a data split of
        (TRAINING, VALIDATION, TESTING) set accordinng to the ratio
    """
    X = data_lst[0]
    N = X.shape[0] # total number of data
    if shuffle:
        idx = np.random.shuffle(np.arange(N))
    else:
        idx = np.arange(N)

    data_lst = [elt[idx, ...][0,...] for elt in data_lst]

    trN = int(tr_ratio*N)
    tr_lst = [elt[:trN,...] for elt in data_lst]

    vtN = int((N - trN)*0.5)
    val_lst = [elt[trN:(trN+vtN),...] for elt in data_lst]
    te_lst  = [elt[trN+vtN:,...] for elt in data_lst]
    return tr_lst, val_lst, te_lst

def img_prep(img_path, target_size = (256, 256)):
    img = cv2.imread(img_path, 0)
    # print ('img before reshping: ', img.shape, 'target size: ', target_size)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    # print ('img after reshaping: ', img.shape)
    return img

def make_aloi_data_loader(size=(128, 128), norm="norm"):
    filenames = os.listdir(PATH_ALOI)

    W, H = size
    # W, H = 768, 576
    W, H = 342, 256
    N = len(filenames)
    # N = 10

    # testing code for total valid files
    ct = 0
    for i in range(N):
        left_path = os.path.join(PATH_ALOI, '%s/%s_l.png'%(filenames[i], filenames[i]))
        right_path = os.path.join(PATH_ALOI, '%s/%s_r.png'%(filenames[i], filenames[i]))

        if not (os.path.isfile(left_path) and os.path.isfile(right_path)):
            continue

        # print ('pre-processing image: ', filenames[i])
        F, pts1, pts2 = get_FMat(left_path, right_path)

        if F is None:
            continue

        ct += 1

    print ('total data: ', N)
    X = np.zeros((ct, H, W, 2)) # H, W interchanged here since numpy takes H,W as input 
    Y = np.zeros((ct, 9))
    P1_lst, P2_lst = [], []
    kpts_cnt = 200 # max 100 kpts
    ind = 0

    for i in range(N):
        left_path = os.path.join(PATH_ALOI, '%s/%s_l.png'%(filenames[i], filenames[i]))
        right_path = os.path.join(PATH_ALOI, '%s/%s_r.png'%(filenames[i], filenames[i]))

        if not (os.path.isfile(left_path) and os.path.isfile(right_path)):
            continue

        print ('pre-processing image: ', filenames[i])
        F, pts1, pts2 = get_FMat(left_path, right_path)

        if F is None:
            continue

        left_img  = img_prep(left_path, target_size=(W,H))
        right_img = img_prep(right_path, target_size=(W,H))

        # print ('left image: ', left_img.shape)
        imgs= np.zeros((2, H, W)) # H, W interchanged here since numpy takes H,W as input
        imgs[0,:,:] = left_img
        imgs[1,:,:] = right_img

        X[ind,:] = np.moveaxis(imgs, [0,1,2], [2,0,1])
        # print ('X shape: ', X[i,:].shape)
        Y[ind,:] = np.resize(F, (1,9))
        # pts1, pts2 = note['kpts']
        kpts_cnt = min(len(pts1), kpts_cnt)
        P1_lst.append(np.array(pts1))
        P2_lst.append(np.array(pts2))
        ind += 1

    X, Y = X.astype(np.float32), Y.astype(np.float32)

    # Normalize F-matrices
    if norm == "abs":
        print("[data loader] Use max abs value to normalize the F-matrix")
        Y = Y / (np.abs(Y).max(axis=1)[:,np.newaxis] + 1e-8)
    elif norm == "norm":
        print("[data loader] Use L2 norm to normalize the F-matrix")
        Y = Y / (np.linalg.norm(Y, axis=1)[:,np.newaxis] + 1e-8)
    elif norm == "last":
        print("[data loader] Use last index to normalize the F-matrix")
        Y = Y / (Y[:,-1].reshape(-1)[np.newaxis,1] + 1e-8)
    else:
        raise Exception("Unrecognized normalization methods:%s"%norm)

    P1_lst = [x[:kpts_cnt,:].reshape((1,kpts_cnt,2)) for x in P1_lst]
    P2_lst = [x[:kpts_cnt,:].reshape((1,kpts_cnt,2)) for x in P2_lst]
    P1, P2 = np.concatenate(P1_lst, axis=0), np.concatenate(P2_lst, axis=0)
    # print(X.shape)
    # print(Y.shape)
    # print(P1.shape)
    # print(P2.shape)

    X /= 255. # Normalize to [0,1]
    return (lambda : data_spliter([X,Y,P1,P2]))

def make_mvs_data_loader(size=(128,128), norm="norm"):
    path_dir = os.listdir(PATH_MVS)
    N = 0
    for path in path_dir:
        filenames = os.listdir(os.path.join(PATH_MVS,path))
        N += len(filenames)

    W, H = size
    W, H = 342, 256
    # N = 10
    X = np.zeros((N, H, W, 2)) # H, W interchanged here since numpy takes H,W as input 
    Y = np.zeros((N, 9))
    P1_lst, P2_lst = [], []
    kpts_cnt = 1000 # max 100 kpts

    i = 0
    for path in path_dir:
        filenames = os.listdir(os.path.join(PATH_MVS, path))
        filenames.sort()

        for j in range(8): #since there are total of 8 different lighting conditions
            left_path = os.path.join(PATH_MVS, path, filenames[j])
            right_path = os.path.join(PATH_MVS, path, filenames[j+8])

            if not (os.path.isfile(left_path) and os.path.isfile(right_path)):
                continue

            print ('pre-processing image: ', left_path)
            F, pts1, pts2 = get_FMat(left_path, right_path)

            if F is None:
                continue

            # print ('F:', F.shape, 'pts1: ', len(pts1), 'pts2: ', len(pts2))

            left_img  = img_prep(left_path, target_size=(W,H))
            right_img = img_prep(right_path, target_size=(W,H))

            # print ('left image: ', left_img.shape)
            imgs= np.zeros((2, H, W)) # H, W interchanged here since numpy takes H,W as input
            imgs[0,:,:] = left_img
            imgs[1,:,:] = right_img

            X[i,:] = np.moveaxis(imgs, [0,1,2], [2,0,1])
            # print ('X shape: ', X[i,:].shape)
            Y[i,:] = np.resize(F, (1,9))
            # pts1, pts2 = note['kpts']
            kpts_cnt = min(len(pts1), kpts_cnt)
            P1_lst.append(np.array(pts1))
            P2_lst.append(np.array(pts2))

            i += 1
            if i >= N:
                break
        
        if i >= N:
            break

    X, Y = X.astype(np.float32), Y.astype(np.float32)
    tot_len = len(P1_lst)
    X, Y = X[:tot_len, :], Y[:tot_len, :]

    # Normalize F-matrices
    if norm == "abs":
        print("[data loader] Use max abs value to normalize the F-matrix")
        Y = Y / (np.abs(Y).max(axis=1)[:,np.newaxis] + 1e-8)
    elif norm == "norm":
        print("[data loader] Use L2 norm to normalize the F-matrix")
        Y = Y / (np.linalg.norm(Y, axis=1)[:,np.newaxis] + 1e-8)
    elif norm == "last":
        print("[data loader] Use last index to normalize the F-matrix")
        Y = Y / (Y[:,-1].reshape(-1)[np.newaxis,1] + 1e-8)
    else:
        raise Exception("Unrecognized normalization methods:%s"%norm)

    P1_lst = [x[:kpts_cnt,:].reshape((1,kpts_cnt,2)) for x in P1_lst]
    P2_lst = [x[:kpts_cnt,:].reshape((1,kpts_cnt,2)) for x in P2_lst]
    P1, P2 = np.concatenate(P1_lst, axis=0), np.concatenate(P2_lst, axis=0)
    # print(X.shape)
    # print(Y.shape)
    # print(P1.shape)
    # print(P2.shape)

    X /= 255. # Normalize to [0,1]
    return (lambda : data_spliter([X,Y,P1,P2]))


def make_kitti_data_loader(size=(128,128), norm="norm"):
    # print ('dataset:',DATASET_PATH)
    path_dir = os.listdir(PATH_CAM)
    print(f"path_dir: {path_dir}")
    '''
    ct = 0
    for path in path_dir:
        path_left = os.path.join(PATH_CAM, path , 'image_00/data/')
        path_right = os.path.join(PATH_CAM, path , 'image_01/data/')

        if not (os.path.isdir(path_left) and os.path.isdir(path_right)):
                continue

        left_filenames = os.listdir(path_left)
        right_filenames = os.listdir(path_right)
        
        for l_path, r_path in zip(left_filenames, right_filenames):

            left_path  = os.path.join(path_left, l_path)
            right_path = os.path.join(path_right, r_path)

            if not (os.path.isfile(left_path) and os.path.isfile(right_path)):
                continue

            # print ('pre-processing image: ', left_path)
            F, pts1, pts2 = get_FMat(left_path, right_path)

            if F is None:
                continue

            ct += 1
    '''
    
    N = 0
    # print ('path_dir: ', path_dir)
    for path in path_dir:
        path_new = os.path.join(PATH_CAM, path, 'image_00/data/')
        if not os.path.isdir(path_new):
            continue
        # print ('path new: ', path_new)
        filenames = os.listdir(path_new)
        N += len(filenames)
    
    W, H = size
    W, H = 1392, 512
    
    # N = len(filenames)
    N = 2000
    print ('N: ', N)
    X = np.zeros((N, H, W, 2)) # H, W interchanged here since numpy takes H,W as input 
    Y = np.zeros((N, 9))
    P1_lst, P2_lst = [], []
    kpts_cnt = 1000 # max 100 kpts
    
    i = 0
    for path in path_dir:
        path_left = os.path.join(PATH_CAM, path , 'image_00/data/')
        print(f"path_left: {path_left}")
        path_right = os.path.join(PATH_CAM, path , 'image_01/data/')
        print(f"path_right: {path_right}")

        if not (os.path.isdir(path_left) and os.path.isdir(path_right)):
                continue

        left_filenames = os.listdir(path_left)
        right_filenames = os.listdir(path_right)
        
        for l_path, r_path in zip(left_filenames, right_filenames):

            left_path  = os.path.join(path_left, l_path)
            right_path = os.path.join(path_right, r_path)

            if not (os.path.isfile(left_path) and os.path.isfile(right_path)):
                continue

            print ('pre-processing image: ', left_path, 'i: ', i)
            F, pts1, pts2 = get_FMat(left_path, right_path)


            if F is None or pts1 is None or pts2 is None:
                continue

            left_img  = img_prep(left_path, target_size=(W,H))
            right_img = img_prep(right_path, target_size=(W,H))

            # print ('left image: ', left_img.shape)
            imgs= np.zeros((2, H, W)) # H, W interchanged here since numpy takes H,W as input
            imgs[0,:,:] = left_img
            imgs[1,:,:] = right_img

            X[i,:] = np.moveaxis(imgs, [0,1,2], [2,0,1])
            # print ('X shape: ', X[i,:].shape)
            Y[i,:] = np.resize(F, (1,9))
            # pts1, pts2 = note['kpts']
            kpts_cnt = min(len(pts1), kpts_cnt)
            P1_lst.append(np.array(pts1))
            P2_lst.append(np.array(pts2))

            i += 1
            if i >= N:
                break

        if i >= N:
                break

    X, Y = X.astype(np.float32), Y.astype(np.float32)
    tot_len = len(P1_lst)
    X, Y = X[:tot_len, :], Y[:tot_len, :]
    
    
    # changed here
    # Normalize F-matrices
    if norm == "abs":
        print("[data loader] Use max abs value to normalize the F-matrix")
        Y = Y / (np.abs(Y).max(axis=1)[:,np.newaxis] + 1e-8)
    elif norm == "norm":
        print("[data loader] Use L2 norm to normalize the F-matrix")
        Y = Y / (np.linalg.norm(Y, axis=1)[:,np.newaxis] + 1e-8)
    elif norm == "last":
        print("[data loader] Use last index to normalize the F-matrix")
        Y = Y / (Y[:,-1].reshape(-1)[np.newaxis,1] + 1e-8)
    else:
        raise Exception("Unrecognized normalization methods:%s"%norm)
    
    P1_lst = [x[:kpts_cnt,:].reshape((1,kpts_cnt,2)) for x in P1_lst]
    P2_lst = [x[:kpts_cnt,:].reshape((1,kpts_cnt,2)) for x in P2_lst]
    P1, P2 = np.concatenate(P1_lst, axis=0), np.concatenate(P2_lst, axis=0)
    # print(X.shape)
    # print(Y.shape)
    # print(P1.shape)
    # print(P2.shape)

    X /= 255. # Normalize to [0,1]
    return (lambda : data_spliter([X,Y,P1,P2]))

'''
def make_syn_data_loader(size=(128,128), data_path=os.path.join(CFD, "data","batch2"),
                         max_num=None, title=None, norm="norm"):
    """
    Takes [data_path], where lives a batch of synthetic data, and return a data
    loader function that could return a tuple of :
        (trX, trY, valX, valY, teX, teY)
    Args:
        [norm]  "norm"|"abs"|"last"
    """
    img_path    = os.path.join(data_path, "img")
    note_path   = os.path.join(data_path, "data_annotations.json")
    notes   = json.load(open(note_path))

    # check whether the image is clean
    clean_fpath = os.path.join(data_path, "clean_file_names.json")
    if os.path.isfile(clean_fpath):
        clean   = json.load(open(clean_fpath))
        def is_clean(left, right):
            return left in clean and right in clean
    else:
        def is_clean(left, right):
            return True

    W, H    = size
    N       = len(notes) if max_num is None else max_num

    # Prepare image names -> image
    img_data = {}
    pbar = progressbar.ProgressBar()  # Progressbar can guess max_value automatically.
    for i in pbar(range(N)):
        note = notes[i]
        left_img_name  = os.path.basename(note['left'])
        right_img_name = os.path.basename(note['right'])
        left_path  = os.path.join(img_path, left_img_name)
        right_path = os.path.join(img_path, right_img_name)
        if not (os.path.isfile(left_path) and os.path.isfile(right_path)):
            continue
        if not is_clean(left_img_name, right_img_name):
            continue

        if str(note['left']) not in img_data:
            img_data[str(note['left'])] = img_prep(left_path,  target_size=(W,H))

        if str(note['right']) not in img_data:
            img_data[str(note['right'])] = img_prep(right_path, target_size=(W,H))

    # discard the last column, which is always 1
    X, Y    = np.zeros((N, W, H, 2)), np.zeros((N, 9))
    P1_lst, P2_lst = [], []
    kpts_cnt = 100 # max 100 kpts
    pbar = progressbar.ProgressBar()  # Progressbar can guess max_value automatically.
    for i in pbar(range(N)):
        note = notes[i]
        if not (str(note['left']) in img_data and str(note['right']) in img_data):
            continue
        left_img  = img_data[str(note['left'])]
        right_img = img_data[str(note['right'])]
        imgs= np.zeros((2, W, H))
        imgs[0,:,:] = left_img
        imgs[1,:,:] = right_img

        X[i,:] = np.moveaxis(imgs, [0,1,2], [2,0,1])
        Y[i,:] = np.resize(np.array(note['fmat']), (1,9))
        pts1, pts2 = note['kpts']
        kpts_cnt = min(len(pts1), kpts_cnt)
        P1_lst.append(np.array(pts1))
        P2_lst.append(np.array(pts2))

    # X, Y = np.moveaxis(X.astype(np.float32), [1,2,3], [3,1,2]), Y.astype(np.float32)
    X, Y = X.astype(np.float32), Y.astype(np.float32)

    # Normalize F-matrices
    if norm == "abs":
        print("[data loader] Use max abs value to normalize the F-matrix")
        Y = Y / (np.abs(Y).max(axis=1)[:,np.newaxis] + 1e-8)
    elif norm == "norm":
        print("[data loader] Use L2 norm to normalize the F-matrix")
        Y = Y / (np.linalg.norm(Y, axis=1)[:,np.newaxis] + 1e-8)
    elif norm == "last":
        print("[data loader] Use last index to normalize the F-matrix")
        Y = Y / (Y[:,-1].reshape(-1)[np.newaxis,1] + 1e-8)
    else:
        raise Exception("Unrecognized normalization methods:%s"%norm)

    P1_lst = [x[:kpts_cnt,:].reshape((1,kpts_cnt,2)) for x in P1_lst]
    P2_lst = [x[:kpts_cnt,:].reshape((1,kpts_cnt,2)) for x in P2_lst]
    P1, P2 = np.concatenate(P1_lst, axis=0), np.concatenate(P2_lst, axis=0)
    # print(X.shape)
    # print(Y.shape)
    # print(P1.shape)
    # print(P2.shape)

    if title is not None:
        print("Saving data to file...")
        pickle.dump([X,Y,P1,P2], open("data/data_%s.pkl" % title, "wb"))

        print("Saving mean data statistics...")
        pickle.dump((np.mean(X), np.mean(Y)), open("data/data_%s.pkl" % title, "wb"))

    # import pdb;pdb.set_trace()
    # X -= np.mean(X)         # normalize to 0 mean
    # X /= np.var(X, axis=0)  # normalize to 1 variance
    X /= 255.               # Normalize to [0,1]
    return (lambda : data_spliter([X,Y,P1,P2]))
'''

def make_data_loader_from_file(file_name):
    print("Loading data from file: %s..." % file_name)
    s = time.time()
    X,Y = pickle.load(open(file_name))
    print("time elapsed:%s" % (time.time() - s))
    return (lambda : data_spliter(X,Y))
'''
if __name__ == "__main__":
    data_loader = make_syn_data_loader(size=(256,256), max_num=10000, title=None)
    # data_loader = make_data_loader_from_file("data/data_4000.pkl")
    tr_lst, val_lst, te_lst = data_loader()
    trX, trY, trP1, trP2 = tr_lst[:4]
    print(trX.shape)
    print(trY.shape)
    print(trP1.shape)
    print(trP2.shape)
    img1 = (trX[0][0] * 256).astype(np.int32)
    img2 = (trX[0][1] * 256).astype(np.int32)
    cv2.imwrite("img1.png", img1)
    cv2.imwrite("img2.png", img2)
'''
if __name__ == '__main__':
    norm = sys.argv[1]
    print ('norm: ', norm)
    from pathlib import Path
    base_path = f'../saved_npy/{norm}'
    Path(base_path).mkdir(exist_ok=True, parents=True)

    data_loader = make_kitti_data_loader(norm=norm)
    tr_lst, val_lst, te_lst = data_loader()

    print ('train: ', tr_lst[0].shape, tr_lst[1].shape, tr_lst[2].shape, tr_lst[3].shape)
    print ('val: ', val_lst[0].shape, val_lst[1].shape, val_lst[2].shape, val_lst[3].shape)
    print ('test: ', te_lst[0].shape, te_lst[1].shape, te_lst[2].shape, te_lst[3].shape)
    
    
    # np.save('../saved_npy/'+norm+'/tr_X.npy',tr_lst[0])
    np.save('../saved_npy/'+norm+'/tr_Y_NN.npy',tr_lst[1])
    # np.save('../saved_npy/'+norm+'/tr_P1.npy',tr_lst[2])
    # np.save('../saved_npy/'+norm+'/tr_P2.npy',tr_lst[3])

    # np.save('../saved_npy/'+norm+'/val_X.npy',val_lst[0])
    np.save('../saved_npy/'+norm+'/val_Y_NN.npy',val_lst[1])
    # np.save('../saved_npy/'+norm+'/val_P1.npy',val_lst[2])
    # np.save('../saved_npy/'+norm+'/val_P2.npy',val_lst[3])

    # np.save('../saved_npy/'+norm+'/te_X.npy',te_lst[0])
    np.save('../saved_npy/'+norm+'/te_Y_NN.npy',te_lst[1])
    # np.save('../saved_npy/'+norm+'/te_P1.npy',te_lst[2])
    # np.save('../saved_npy/'+norm+'/te_P2.npy',te_lst[3])
    
    # tr_lst = np.load('../saved_npy/norm/tr.npz')
    # val_lst = np.load('../saved_npy/norm/val.npz')
    # te_lst = np.load('../saved_npy/norm/te.npz')