"""
This file contributes to generate povray dataset
"""

import os
import json
import random
import progressbar
import numpy as np
import pickle as pkl
CFD = os.path.dirname(os.path.realpath(__file__))

PI_R = [10, 10, 10] # Maximu rotateion
PI_T = [1, 1, 1] # Maximum translate

def inject_camera_rt(povfile, output=None, mode="FORMATTED", piR = PI_R, piT = PI_T):
    print ("Hijecking file %s " % povfile)
    r1, r2 = np.random.rand(3), np.random.rand(3)
    r1, r2 = r1/np.sum(r1), r2 / np.sum(r2)
    rotate = piR * (r1 - np.ones(3) * 0.5) * 2
    transl = piT * (r2 - np.ones(3) * 0.5) * 2
    focal  = np.random.rand(1)[0] * 1 + 1
    fdict  = {
        "focal"     : focal,
        "rotate"    : "<%s, %s, %s>" % (rotate[0], rotate[1], rotate[2]),
        "translate" : "<%s, %s, %s>" % (transl[0], transl[1], transl[2])
    }
    print ("\tParameters: R=%s;\tT=%s" % (rotate, transl))

    # TODO: deal with output
    if output is None:
        output = "%s_r%s_t%s" % (rotate, transl)
    print ("Outputing povfile: %s" % output)

    if mode is "FORMATTED":
        with open(povfile) as fin, open(output, 'wb') as fout:
            fout.write(''.join(fin.readlines()) % fdict)
    else:
        pass # TODO need to find the camera parameter and injected into it

    return {
        "focal"     : focal,
        "rotate"    : rotate.tolist(),
        "transl"    : transl.tolist(),
        "povfile"   : output
    }

def gen_group(povfile, group_dir, gname, ttl = 10, mode = "FORMATTED"):
    annotations = []
    print ("Generating povray groups")
    for i in range(ttl):
        print ("[%s/%s]:" % (i, ttl))
        output = os.path.join(group_dir, "gen_%s_%s.pov" % (gname, i))
        config = inject_camera_rt(povfile, output=output, mode=mode)
        annotations.append(config)
    return annotations

def render_group(gdir, gname, annotes, out_dir=os.path.join(CFD, "output"), w=512, h=512):
    img_notes = []
    count, ttl = 0, len(annotes)
    for config in annotes:
        count+=1
        pov = config['povfile']
        out_img  = os.path.join(out_dir, "%s_%s.png" % (gname, os.path.basename(pov)))
        cmd = "cd %s;" % gdir + \
            " povray " + \
            " %s " % pov + \
            " +o\"%s\" " % out_img + \
            " +W%s +H%s > /dev/null 2>&1" % (w,h)
        print ("[%s/%s] : %s" % (count, ttl, cmd ))
        os.system(cmd)
        img_config = dict(config)
        img_config.update({ 'img_path' : out_img })
        img_notes.append(img_config)
    return img_notes

if __name__ == "__main__":
    # gname = "cleaning"
    gname_t = "pomme_t10_r1_%s"
    big_config = []
    for i in range(6):
        gname = gname_t % i
        gdir = os.path.join(CFD, "pomme")
        povfile = os.path.join(gdir, "template%s.pov" % i)
        notes = gen_group(povfile, gdir, gname, ttl = 100)
        config = render_group(gdir, gname, notes)
        json.dump(config, open("%s.json" % gname, "wb"))
        big_config += config
    json.dump(big_config, open("pomme.json", "wb"))
