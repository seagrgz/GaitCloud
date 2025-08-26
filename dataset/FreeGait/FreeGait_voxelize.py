#!/usr/bin/env python
import numpy as np
import os
import sys
import time
import pickle
import math
import yaml

sys.path.append('../../util/')
from pretreatment import seq_voxelize

data_root = 'FreeGait'
dst = 'FreeGait-voxel.20tempspat20centcomp'
#name = 'test'
frame_num = 20
box_size = [1.25,1.25,2]
res = 0.03125
compress = True
tfusion = 'spat+tmp' #tmp, spat, tmp+spat
tmp_diff = False
dilution = 1
noise = 0

def FreeGait_voxelize(data_root,
        dst, 
        frame_num, 
        box_size,
        res,
        dilution=1, 
        noise=0, 
        counter=1, 
        compress=0,
        tfusion=False,
        tmp_diff=False):

    if isinstance(res, float):
        res = [res, res, res]
    elif isinstance(res, list):
        pass
    else:
        raise NotImplementedError('res should be list or float')

    if os.path.exists(dst):
        print('Removing old data ...')
        os.system('rm -rf {}/**'.format(dst))
    else:
        os.system('mkdir {}'.format(dst))

    target_list = sorted(os.listdir(data_root)) 
    target_num = len(target_list)
    st = time.time()
    for i, target in enumerate(target_list):
        dev_root = os.path.join(data_root, target)
        devs = os.listdir(dev_root)
        for dev in devs:
            seq_root = os.path.join(dev_root, dev)
            seq_names = os.listdir(seq_root)
            for seq in seq_names:
                sample_path = os.path.join(seq_root, seq, 'lidar.pkl')
                namespace = '-'.join([target, dev, seq])
                if os.path.exists(sample_path):
                    with open(sample_path, 'rb') as f:
                        data = pickle.load(f)
                    f.close()
                else:
                    print('Warning: {} not found!'.format(sample_path))
                    continue

                flag = seq_voxelize(data, dst, namespace, frame_num, box_size, res, dilution, noise, counter, compress, tfusion, tmp_diff)
                if flag:
                    print('Warning: Target {}~{}~{} discard'.format(target, attr, vp))

        t_remain = (time.time()-st)/(i+1) * (target_num-i-1)
        print('Target {} completed, time remain: {}h-{}m-{}s'.format(target, int(t_remain//3600), int(t_remain%3600//60), int(t_remain%60)))

if __name__ == '__main__':
    #for noise in [0.005,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085,0.095]:
    #    voxelize('/home/sx-zhang/SUSTech1K/SUSTech1K-Released-voxel.20dev{}'.format(noise), frame_num=20, dilution=1, noise=noise)
    FreeGait_voxelize(data_root = data_root,
            dst = dst,
            frame_num = frame_num,
            box_size = box_size,
            res = res,
            dilution = dilution,
            noise = noise,
            compress = compress,
            tfusion = tfusion,
            tmp_diff = tmp_diff)
