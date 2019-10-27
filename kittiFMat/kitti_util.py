import os
import sys
import json
import itertools
import numpy as np
from parser import KittiParamParser
from collections import defaultdict

calib_links = [
    "http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_calib.zip",
    "http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_28_calib.zip",
    "http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_29_calib.zip",
    "http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_30_calib.zip",
    "http://kitti.is.tue.mpg.de/kitti/raw_data/2011_10_03_calib.zip"
]

def dump_all_fmats(outfile = "kittis_all_fmat.json"):
    fmats = {}
    for l in calib_links:
	l = l.strip()
	zip_fname = os.path.basename(l)
	base_name  = zip_fname[:-4]
	date = base_name[:-6]
	print (zip_fname, base_name, date)
	os.system("rm -rf %s" % date)
	os.system("wget %s" % l)
	os.system("unzip %s" % zip_fname)
	os.system("rm %s" % zip_fname)
	fmats[date] = defaultdict(lambda : defaultdict(lambda : np.zeros((3,3))))
	p = KittiParamParser(os.path.join(date, "calib_cam_to_cam.txt"))
	for i,j in itertools.combinations(range(4), 2):
            fmats[date][i][j] = p.get_F(i,j).tolist()
            fmats[date][j][i] = p.get_F(j,i).tolist()
	os.system("rm -rf %s" % date)
    fmats = dict(fmats)
    json.dump(fmats,open(outfile, "w+"))
    return fmats

if __name__ == "__main__":
    dump_all_fmats()
