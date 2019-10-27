# evaluating evaluation metrics
import numpy as np
from matplotlib import pyplot as plt
# def perturb_F(F, i, j, delta):
#     '''for now, do stupid, not-make-sense, element-wise 
#     perturbation.'''
#     assert F.shape == (3,3), F.size
#     dF = np.zeros((3,3))
#     dF[i,j] = delta
#     return [F + x * dF for x in range(100)]

def get_K(fx, fy, s, cx, cy):
    return np.array([
            [fx,s,cx],
            [0,fy,cy],
            [0,0,1]
            ])

def get_F(params):
    '''
    params = dictionary of params
    '''
    a,b,c = params['alpha'],params['beta'],params['gamma']
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(a),np.sin(a)],
        [0,-np.sin(a),np.cos(a)]
        ])
    Ry = np.array([
        [np.cos(b), 0, -np.sin(b)],
        [0, 1, 0],
        [np.sin(b),0,np.cos(b)]
        ])
    Rz = np.array([
        [np.cos(c), np.sin(c), 0],
        [-np.sin(c), np.cos(c),0],
        [0, 0, 1]
        ])
    R = Rx.dot(Ry.dot(Rz))
    tx, ty, tz = params['tx'], params['ty'], params['tz']
    T = np.array([
            [0, -tz, ty],
            [tz, 0, -tx],
            [-ty, tx, 0]
        ])
    K1 = get_K(params['fx1'], params['fy1'], params['s1'], params['cx1'], params['cy1'])
    K2 = get_K(params['fx2'], params['fy2'], params['s2'], params['cx2'], params['cy2'])
    return np.linalg.inv(K2.T).dot(R.dot(T.dot(np.linalg.inv(K1))))

def perturb_F(params, pertb, N = 500, step = 0.01):
    '''pertb is the name of the parameter to perturb'''
    out = []
    F_true = get_F(params)
    ptb_params = dict(params)
    for i in range(N):
        ptb_params[pertb] += i * step
        out.append(get_F(ptb_params))
    return out

# def perturb_F_rotation(theta):
#     t = np.pi / 3
#     return [get_F(t + np.pi / 100 * x, default_trans, K1, K2) for x in range(100)]

# def perturb_F_translation(delta = 0.1):
#     theta = np.pi / 3
#     return [get_F(theta, default_trans + x * delta, K1, K2) for x in range(100)]

# def perturb_F(R, t, K1, K2, i, delta):


def frobenius_metric(f, true_f):
    return np.linalg.norm(f-true_f)

def eval_metric(g, params):
    '''g is a metric : FxF->R'''
    true_F = get_F(params)
    for k in params.keys():
        Fs = perturb_F(params, k)
        ds = [frobenius_metric(F,true_F) for F in Fs ]
        plt.figure()
        plt.plot(ds)
        plt.title(k)
        plt.savefig('figs/{}'.format(k))


# def hough_transform(F,img1, img2):

def gen_pair(K1, K2, R, T, p):
    P = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
            ])
    p1_homo = K1.dot(P.dot(np.append(p,1)))
    p2_homo = K2.dot(P.dot(R.dot(T.dot(np.append(p,1)))))
    p1 = p1_homo[:2]/p1_homo[2]
    p2 = p2_homo[:2]/p2_homo[2]
    return p1, p2









if __name__ == '__main__':
    params = {
        'alpha' : 0,
        'beta'  : 0,
        'gamma' : 0,
        'tx'    : 1,
        'ty'    : 1,
        'tz'    : 1,
        'fx1'   : 1,
        'fy1'   : 1,
        's1'    : 1,
        'cx1'   : 1,
        'cy1'   : 1,
        'fx2'   : 1,
        'fy2'   : 1,
        's2'    : 1,
        'cx2'   : 1,
        'cy2'   : 1,
    }
    eval_metric(frobenius_metric, params)


