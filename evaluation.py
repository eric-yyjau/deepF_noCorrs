import numpy as np
from povray import stereo_pair

class KPCorrBasedMetric(object):
    """Assuming 8-entries regression"""
    def __init__(self, metric, normalize=False, epsilon=1e-5):
        self.metric = metric
        self.normalize = normalize
        self.epsilon = epsilon

    def __call__(self, f_pred, f_gtr, pts1, pts2):
        scores = []
        scores_pred = []
        scores_gtrs = []
        for i in range(f_pred.shape[0]):
            pred_fmat = np.ones(9)
            gtrs_fmat = np.ones(9)
            pred_fmat[:9] = f_pred[i,...]
            gtrs_fmat[:9] = f_gtr[i, ...]
            # kpts1 = list([list(x) for x in pts1[i,...]])
            # kpts2 = list([list(x) for x in pts2[i,...]])
            kpts1 = pts1[i,...]
            kpts2 = pts2[i,...]
            err_pred = self.metric(pred_fmat.reshape((3,3)), kpts1, kpts2)
            err_gtrs = self.metric(gtrs_fmat.reshape((3,3)), kpts1, kpts2)
            scores_pred.append(err_pred)
            scores_gtrs.append(err_gtrs)
            if self.normalize:
                scores.append((err_pred - err_gtrs)/(err_gtrs + self.epsilon))
            else:
                scores.append(err_pred - err_gtrs)

        scores = np.array(scores)
        scores_pred = np.array(scores_pred)
        scores_gtrs = np.array(scores_gtrs)
        return scores, scores_pred, scores_gtrs

sampson_dist = KPCorrBasedMetric(stereo_pair.sampson_dist)
epipolar_constraint = KPCorrBasedMetric(stereo_pair.err_pts_correspond)
epipolar_constraint_abs = KPCorrBasedMetric(stereo_pair.abserr_pts_correspond)
sym_epilolar_dist = KPCorrBasedMetric(stereo_pair.sym_epipolar_dist)

if __name__ == "__main__":
    import data_loader
    tr_ds, val_ds, te_ds = data_loader.make_povray_datasets(max_num=2000)
    for bs in range(10):
        batch_size = int(2**bs)
        for sigma in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
            metrics = {
                "sampson_dist  " : sampson_dist,
                "lin_reconstuct" : linear_reconstruction_err,
                "epi_constraint" : epipolar_constraint
            }
            scores =  { k:0 for k in metrics.keys() }
            s_vars =  { k:0 for k in metrics.keys() }
            num_batches = len(tr_ds)//batch_size+1
            for i in range(num_batches):
                bx, by, bp1, bp2 = val_ds(batch_size)
                for k in metrics.keys():
                    m = metrics[k]
                    p_by = by+np.random.rand(*by.shape)*sigma
                    m_score= np.array(m(p_by, by, bp1, bp2))
                    scores[k] += m_score.mean()
                    s_vars[k] += m_score.var()
            print("Perterbation:%.7f\tBatchsize=%d"%(sigma, batch_size))
            for k in scores.keys():
                s = scores[k]
                v = s_vars[k]
                print("\t%s\t%.5f[%.5f]"%(k, s/float(num_batches), v/float(num_batches)))

