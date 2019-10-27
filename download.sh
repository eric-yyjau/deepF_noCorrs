#! /bin/bash

# This script is used to download the Kitti dataset

DATA_DIR="data/kitti"

mkdir -p $DATA_DIR

base_url="http://kitti.is.tue.mpg.de/kitti/raw_data/"

#For now just download one dataset
date="2011_09_26"
ds_name="${date}_drive_0001"

imgs_url="http://kitti.is.tue.mpg.de/kitti/raw_data/${ds_name}/${ds_name}_sync.zip"
calib_url="http://kitti.is.tue.mpg.de/kitti/raw_data/${date}_calib.zip"
# trklet_url="http://kitti.is.tue.mpg.de/kitti/raw_data/${ds_name}/${ds_name}_tracklets.zip"

wget $imgs_url -O "$DATA_DIR/${ds_name}.zip" 
wget $calib_url -O "$DATA_DIR/${date}_calib.zip"

unzip "$DATA_DIR/${ds_name}.zip" 
unzip "$DATA_DIR/${date}_calib.zip"