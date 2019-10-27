#! /bin/bash

for f in ../data/kitti/2011_09_26/2011_09_26_drive_0001_sync/image_00/data/*; do
	f=`basename $f`
	echo $f
	python kitti_fmat.py --r 0.2 --img1 `pwd`/../data/kitti/2011_09_26/2011_09_26_drive_0001_sync/image_00/data/$f --img2 `pwd`/../data/kitti/2011_09_26/2011_09_26_drive_0001_sync/image_01/data/$f || exit 1
done
