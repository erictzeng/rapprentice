#!/usr/bin/env python

import pdb
import numpy as np
import h5py
from rapprentice import clouds

DS_SIZE = .025

def get_downsampled_clouds(demofile):
    return [clouds.downsample(seg["cloud_xyz"], DS_SIZE) for seg in demofile.values()]

def main():
    demofile = h5py.File('/home/shhuang/research-lfd/data/overhand/all.h5', 'r')
    keys = demofile.keys()

    # Get point clouds of initial scenes
    ds_clouds = get_downsampled_clouds(demofile)

    # Calculate features in frame of gripper (e.g. shape context)
    for key in demofile:
        seg_info = demofile[key]
        for lr in 'lr':
            link_name = "%s_gripper_tool_frame"%lr
            traj_hmat = np.asarray(seg_info[link_name]["hmat"])

if __name__ == '__main__':
    main()
