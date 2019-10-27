"""
This file aimed to provide pipeline functions to make cascade data
"""

import os
import csv
import random
import util as syn_util
from scipy import misc
from progressbar import ProgressBar
FD = os.path.dirname(os.path.realpath(__file__))

ORIENT = [ "0,0", "90,0", "0,90", "180,0" ]
MIN_X  = 10.0
MIN_Y  = 8.0
MIN_Z  = 0.0

X_RANGE = 45.0
Y_RANGE = 16.0
Z_RANGE = 360.0

# TODO: this function has been defined twice already
def frange(start, end, step):
    """
    Range funciton for floats
    """
    ret  = []
    curr = start
    while curr < end:
        ret.append(curr)
        curr += step
    return ret

def gen_image_with_bg(pov_lst, bg_map, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    imgs = []
    bg_info = {}
    for pov in pov_lst:
        pov_name = os.path.basename(pov)
        out_img  = os.path.join(out_dir, "%s.png" % pov_name)
        cmd = "povray " + \
            " %s " % pov + \
            " +L'lgeo/lg' " + \
            " +o%s " % out_img + \
            " +W400 +H300 +Q11 +A +R6 > /dev/null 2>&1"
        print (cmd)
        os.system(cmd)

        imgs.append(out_img)
        bg_info[out_img] = bg_map[pov]
    return imgs, bg_info

def shift_images(img_lst, bg_map, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ret     = []
    bg_info = {}
    for f in img_lst:
        # if not f.endswith('png') or f.startswith("."): continue
        print (f, bg_map[f])
        fg = misc.imread(f)[...,:3]
        bg = misc.imread(bg_map[f])[...,:3]

        print (fg.shape, bg.shape)

        fname = os.path.basename(f)
        out_f = os.path.join(out_dir, fname)
        print (fname, out_f)
        misc.imsave(out_f, syn_util.shift(bg,fg))

        ret.append(out_f)
        bg_info[out_f] = bg_map[f]
    return ret, bg_info

def gen_bgless_imgs(pov_lst, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    imgs = []
    for pov in pov_lst:
        pov_name = os.path.basename(pov)
        out_img  = os.path.join(out_dir, "%s.png" % pov_name)
        cmd = "povray " + \
            " %s " % pov + \
            " +L'lgeo/lg' " + \
            " +o%s " % out_img + \
            " +W400 +H300 +Q11 +A +UA > /dev/null 2>&1"
        print (cmd)
        os.system(cmd)
        imgs.append(out_img)

    return imgs

def make_bbox(imgs, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ret = []
    dat_file   = open(os.path.join(out_dir, 'info.dat'), 'w+')
    dat_writer = csv.writer(dat_file, delimiter='\t')
    for f in imgs:
        if not f.endswith('png') or f.startswith("."): continue
        bbox = syn_util.get_bbox(f)
        dat_writer.writerow([os.path.basename(f), 1, "%s %s %s %s" % bbox ])
        ret.append(bbox)
    dat_file.close()
    return ret

def make_pov(template_path, params):
    template = ''.join(open(os.path.join(FD, template_path)).readlines())
    for k,v in params.iteritems():
        template = template.replace(str(k),str(v))
    return template

def make_pov_batch(bg_dir, size_param, lego_ids, out_dir):
    """
    Generate .pov
    """

    if not os.path.exists("%s_bbox" % out_dir):
        os.makedirs("%s_bbox" % out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Make configurations
    # size_param = max(2, size_param)
    size_param = max(1, size_param)
    x_values = frange(MIN_X, MIN_X + X_RANGE, X_RANGE / float(size_param))
    y_values = frange(MIN_Y, MIN_Y + Y_RANGE, Y_RANGE / float(size_param))
    centers  = [ (x, y) for x in x_values for y in y_values ]

    z_values = frange(MIN_Z, MIN_Z + Z_RANGE, Z_RANGE / float(size_param))
    rotates  = [ (o, z) for o in ORIENT for z in z_values ]

    conf = [ (x,y,o,z) for (x,y) in centers for (o,z) in rotates ]

    bgs  = [ os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.endswith("png")]
    ret  = []
    bg_map = {}
    bgless_ret = []
    for bid in lego_ids:
        for x,y,o,z in conf:
            curr_bg  = random.choice(bgs)
            params   = {
                    'MODEL_ID'          : bid,
                    "DUMMY_BG_IMG"      : curr_bg,
                    'DUMMY_TRANSLATE'   : "<%s,%s,0>" % (x,y),
                    "DUMMY_ROTATE"      : "<%s,%s>"   % (o,z)
            }
            template = make_pov('template.pov', params)

            out_fname = os.path.join(out_dir, "%s_c%s-%s-0_r%s-%s.pov" % (bid,x,y,o,z))
            out_file  = open(out_fname, 'w+')
            out_file.write(template)
            out_file.close()

            # Make backgroundless pov
            bgless_params = {
                    'MODEL_ID'          : bid,
                    'DUMMY_TRANSLATE'   : "<%s,%s,0>" % (x,y),
                    "DUMMY_ROTATE"      : "<%s,%s>"   % (o,z)
            }
            template_bgless = make_pov('template_bbox.pov', bgless_params)

            bgless_out_fname = os.path.join("%s_bbox" % out_dir,
                        "%s_c%s-%s-0_r%s-%s.pov" % (bid,x,y,o,z))
            out_file  = open(bgless_out_fname, 'w+')
            out_file.write(template_bgless)
            out_file.close()

            ret.append(out_fname)
            bg_map[out_fname] = curr_bg
            bgless_ret.append(bgless_out_fname)

    return ret, bg_map, bgless_ret

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_num',  type=str, default="3",
                        help="The batch number of the data generated")
    parser.add_argument('--size',       type=int, default=4,
                        help="The size parameters of the generated dat.")
    parser.add_argument('--bg_dir',     type=str, default=None,
                        help="The background directory for the negative data.")

    args = parser.parse_args()
    # Just a stub
    bg_dir = "/Users/Grendel/Dropbox/Osmo-Lego/data/neg"

    BIDS  = [ 3001, 3037, 2456, 2877, 3002, 3003,  3004, 3005, 3010, 30363 ]
    # BIDS = [ 3001 ]
    batch = args.batch_num

    povs, bg_map, bgless_pov = make_pov_batch(bg_dir, args.size, BIDS, "b%s_bg_pov" % batch)
    imgs, bg_map = gen_image_with_bg(povs, bg_map, "b%s_bg_img" % batch)
    imgs, bg_map = shift_images(imgs, bg_map, "b%s_bg_shift_img" % batch)
    imgs = gen_bgless_imgs(bgless_pov, "b%s_bgless_img" % batch)
    bbox = make_bbox(imgs, "b%s_bbox" % batch)
