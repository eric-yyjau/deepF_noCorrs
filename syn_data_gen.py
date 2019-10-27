"""
Main interface to generate synthetic data
"""
import os
import cv2
import util
import json
import itertools
from povray import dat_gen
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count

def compute_fmat(config, i, j, verbose = False):
    conf1, conf2 = config[i], config[j]
    file1, file2 = conf1['img_path'], conf2['img_path']
    if verbose: print ("Generating stero pair from %s to %s" % (file1, file2))
    img1, img2 = cv2.imread(file1, 0), cv2.imread(file2, 0)
    r = 0.5
    total = 50

    if verbose: print ("Computing using LEMED...")
    try:
        F, pts1, pts2 = util.sift_get_fmat(img1, img2, total = total, ratio = r, display=False)
        all_pts1, all_pts2 = pts1 , pts2
    except:
        if verbose: print ("Exception encountered while generating F-mat")
        return {"except":"Occured"}

    if verbose: print ("F-matrix (from sift LEMED): \n%s" % F)
    err = util.err_pts_correspond(F, all_pts1, all_pts2)
    if verbose: print ("pFp avg sum err: %s " % err)
    if abs(err) > 1e-3:
        if verbose: print ("Skipping the data since error is too high")
        return {"error": err }

    return {
        "left"  : file1,
        "right" : file2,
        "r"     : r,
        "kpts"  : [pts1.tolist(), pts2.tolist()],
        "fmat"  : F.tolist(),
        "rotate_left"  : conf1['rotate'],
        "rotate_right" : conf1['rotate'],
        "transl_left"  : conf1["transl"],
        "transl_right" : conf2["transl"]
    }

def parallel_main(config, all_pairs):
    threads = cpu_count()
    pool = ThreadPool(threads)
    allconfigs = pool.map(
            lambda tpl: compute_fmat(config, tpl[0], tpl[1], verbose = True), all_pairs)
    pool.close()
    pool.join()
    return allconfigs

def refined_main(config, all_pairs):
    allconfigs = map(lambda tpl: compute_fmat(config, tpl[0], tpl[1], verbose = True), all_pairs)
    return allconfigs

if __name__ ==  "__main__":
    config = json.load(open("povray/pomme.json"))
    notes = []
    count_skip, count_err, count_ttl = 0, 0, 0
    all_pairs  = list(itertools.combinations(range(len(config)), 2))

    allconfigs = parallel_main(config, all_pairs)

    good_config= filter(lambda cfg: "except" not in cfg and "error" not in cfg, allconfigs)
    err_config = filter(lambda cfg: "error" in cfg, allconfigs)
    exp_config = filter(lambda cfg: "except" in cfg, allconfigs)

    count_gain = len(good_config)
    count_err  = len(exp_config)
    count_good = len(good_config)
    count_skip = len(err_config)

    print ("Total:%s\tgain:%s\tskip:%s\terror:%s" % (count_ttl, count_gain, count_skip, count_err))
    json.dump(good_config, open("data_annotations.json", "wb"))


