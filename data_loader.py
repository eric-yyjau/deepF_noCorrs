# Data loader for loading both synthetic data and the kitti Dataset

"""
Data Loaders all return a tuple (img1_batch, img2_batch, f_matrix batch)
"""
import os
import data_util
import numpy as np
import nsml
from nsml import DATASET_PATH
# CFD = os.path.dirname(os.path.realpath(__file__))

class DataSet(object):
    """TODO"""
    def __init__(self, X, Y, P1, P2, shuffle=True):
        self.X = X
        self.Y = Y
        self.P1 = P1
        self.P2 = P2
        print("Dataloader:X=%s Y=%s P1=%s P2=%s"\
             %(self.X.shape, self.Y.shape, self.P1.shape, self.P2.shape))
        self.shuffle = shuffle
        self.N = self.X.shape[0]

        self.curr_idx = 0
        self.idx_lst = np.arange(self.N)
        if self.shuffle:
            np.random.shuffle(self.idx_lst)

    def __call__(self, batch_size=32):
        if self.curr_idx + batch_size > self.N:
            np.random.shuffle(self.idx_lst)
            self.curr_idx = 0
        idxs = self.idx_lst[self.curr_idx:self.curr_idx+batch_size]
        bx  = self.X[idxs,...]
        by  = self.Y[idxs,...]
        bp1 = self.P1[idxs,...]
        bp2 = self.P2[idxs,...]
        self.curr_idx += batch_size
        return bx, by, bp1, bp2

    def img_shape(self):
        return self.X.shape[1:]

    def fmat_shape(self):
        return self.Y.shape[1:]

    def __len__(self):
        return self.N

def make_data_loader(norm='norm'):
    path_dir = os.path.join(DATASET_PATH, 'train', norm)

    tr_X = np.load(os.path.join(path_dir, 'tr_X.npy'))
    # tr_Y = np.load(os.path.join(path_dir, 'tr_Y.npy'))
    tr_Y = np.load(os.path.join('saved_npy/norm', 'tr_Y_N.npy'))
    tr_P1 = np.load(os.path.join(path_dir, 'tr_P1.npy'))
    tr_P2 = np.load(os.path.join(path_dir, 'tr_P2.npy'))

    val_X = np.load(os.path.join(path_dir, 'val_X.npy'))
    # val_Y = np.load(os.path.join(path_dir, 'val_Y.npy'))
    val_Y = np.load(os.path.join('saved_npy/norm', 'val_Y_N.npy'))
    val_P1 = np.load(os.path.join(path_dir, 'val_P1.npy'))
    val_P2 = np.load(os.path.join(path_dir, 'val_P2.npy'))

    te_X = np.load(os.path.join(path_dir, 'te_X.npy'))
    # te_Y = np.load(os.path.join(path_dir, 'te_Y.npy'))
    te_Y = np.load(os.path.join('saved_npy/norm', 'te_Y_N.npy'))
    te_P1 = np.load(os.path.join(path_dir, 'te_P1.npy'))
    te_P2 = np.load(os.path.join(path_dir, 'te_P2.npy'))
    
    # Normalize F-matrices
    if norm == "abs":
        print("[data loader] Use max abs value to normalize the F-matrix")
        tr_Y = tr_Y / (np.abs(tr_Y).max(axis=1)[:,np.newaxis] + 1e-8)
        val_Y = val_Y / (np.abs(val_Y).max(axis=1)[:,np.newaxis] + 1e-8)
        te_Y = te_Y / (np.abs(te_Y).max(axis=1)[:,np.newaxis] + 1e-8)
    elif norm == "norm":
        print("[data loader] Use L2 norm to normalize the F-matrix")
        tr_Y = tr_Y / (np.linalg.norm(tr_Y, axis=1)[:,np.newaxis] + 1e-8)
        val_Y = val_Y / (np.linalg.norm(val_Y, axis=1)[:,np.newaxis] + 1e-8)
        te_Y = te_Y / (np.linalg.norm(te_Y, axis=1)[:,np.newaxis] + 1e-8)
    elif norm == "last":
        print("[data loader] Use last index to normalize the F-matrix")
        tr_Y = tr_Y / (tr_Y[:,-1].reshape(-1)[np.newaxis,1] + 1e-8)
        val_Y = val_Y / (val_Y[:,-1].reshape(-1)[np.newaxis,1] + 1e-8)
        te_Y = te_Y / (te_Y[:,-1].reshape(-1)[np.newaxis,1] + 1e-8)
    else:
        raise Exception("Unrecognized normalization methods:%s"%norm)
    
    tr_lst = [tr_X, tr_Y, tr_P1, tr_P2]
    val_lst = [val_X, val_Y, val_P1, val_P2]
    te_lst = [te_X, te_Y, te_P1, te_P2]

    print ('train: ', tr_lst[0].shape, tr_lst[1].shape, tr_lst[2].shape, tr_lst[3].shape)
    print ('val: ', val_lst[0].shape, val_lst[1].shape, val_lst[2].shape, val_lst[3].shape)
    print ('test: ', te_lst[0].shape, te_lst[1].shape, te_lst[2].shape, te_lst[3].shape)

    return tr_lst, val_lst, te_lst

def make_kitti_datasets(size=(256,256), norm='norm'):
    # data_loader = data_util.make_kitti_data_loader(size, norm)
    tr_lst, val_lst, te_lst = make_data_loader(norm)

    trX, trY, trP1, trP2 = tr_lst
    tr_data_loader = DataSet(trX, trY, trP1, trP2)

    valX, valY, valP1, valP2 = val_lst
    val_data_loader = DataSet(valX, valY, valP1, valP2)

    teX, teY, teP1, teP2 = te_lst
    te_data_loader =DataSet(teX, teY, teP1, teP2)

    return tr_data_loader, val_data_loader, te_data_loader

def make_mvs_datasets(size=(256,256), norm='norm'):
    # data_loader = data_util.make_mvs_data_loader(size, norm)
    tr_lst, val_lst, te_lst = make_data_loader()

    trX, trY, trP1, trP2 = tr_lst
    tr_data_loader = DataSet(trX, trY, trP1, trP2)

    valX, valY, valP1, valP2 = val_lst
    val_data_loader = DataSet(valX, valY, valP1, valP2)

    teX, teY, teP1, teP2 = te_lst
    te_data_loader =DataSet(teX, teY, teP1, teP2)

    return tr_data_loader, val_data_loader, te_data_loader    

def make_aloi_datasets(size=(256,256), norm='norm'):
    # data_loader = data_util.make_aloi_data_loader(size, norm)
    tr_lst, val_lst, te_lst = make_data_loader()

    trX, trY, trP1, trP2 = tr_lst
    tr_data_loader = DataSet(trX, trY, trP1, trP2)

    valX, valY, valP1, valP2 = val_lst
    val_data_loader = DataSet(valX, valY, valP1, valP2)

    teX, teY, teP1, teP2 = te_lst
    te_data_loader =DataSet(teX, teY, teP1, teP2)

    return tr_data_loader, val_data_loader, te_data_loader
'''
def make_povray_datasets(data_path=os.path.join(CFD, "data","batch2"),
                         size=(256,256), max_num=None, title=None, norm='norm'):
    data_loader = data_util.make_syn_data_loader(size, data_path, max_num, title, norm)
    tr_lst, val_lst, te_lst = data_loader()

    trX, trY, trP1, trP2 = tr_lst
    tr_data_loader = DataSet(trX, trY, trP1, trP2)

    valX, valY, valP1, valP2 = val_lst
    val_data_loader = DataSet(valX, valY, valP1, valP2)

    teX, teY, teP1, teP2 = te_lst
    te_data_loader = DataSet(teX, teY, teP1, teP2)

    return tr_data_loader, val_data_loader, te_data_loader
'''
if __name__ == "__main__":
    tr, val, te = make_kitti_datasets()

