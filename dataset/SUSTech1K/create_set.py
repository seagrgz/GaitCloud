#!/usr/bin/env python

import os
import json
import yaml
import pickle
import time
import itertools
import random
import numpy as np

data_root = 'SUSTech1K-voxel.2tempspat2centcomp'
lidargait = True
#fname = '01-{}-LiDAR-PCDs_depths.pkl'
#fname = '09-sync-{}-LiDAR-PCDs_depths.pkl'
fname = '14-sync-{}-Camera-Sils_aligned.pkl'
                    
def move_data(root, dst, partition):
    train_set = partition["TRAIN_SET"]
    test_set = partition["TEST_SET"]

    #use all samples
    data_list = sorted(os.listdir(root))
    sample_num = len(data_list) - 1
    unused = []
    st = time.time()
    for i, name in enumerate(data_list):
        if '.npz' in name:
            try:
                target, var, view = name[:-4].split('_')
            except Exception as e:
                print(e)
                raise RuntimeError('Unexpect filename ({}) in root folder'.format(name))
        else:
            continue
        #train set
        if target in train_set:
            os.system('cp {}/{} {}/train/{}'.format(root, name, dst, name))
        #test set
        elif target in test_set:
            os.system('cp {}/{} {}/test/{}'.format(root, name, dst, name))
        else:
            unused.append(name)

        if i%100==0:
            t_remain = (time.time()-st)/(i+1) * (sample_num-i-1)
            print('Processing...{}%, remain time: {}h-{}m-{}s'.format(round(i*100/sample_num,2), int(t_remain//3600), int(t_remain%3600//60), int(t_remain%60)))
    print('Unused samples: ', unused)

def create_raw(data_root, dst, fname, shape, partition):
    train_samples = partition["TRAIN_SET"]
    test_samples = partition["TEST_SET"]
    label_list = sorted(os.listdir(data_root))
    train_set = [label for label in train_samples if label in label_list]
    test_set = [label for label in test_samples if label in label_list]

    target_num = len(label_list)
    unused = []
    st = time.time()
    for i, target in enumerate(label_list):
        attr_root = os.path.join(data_root, target)
        attributes = os.listdir(attr_root)
        for item in attributes:
            sample_root = os.path.join(attr_root, item)
            views = os.listdir(sample_root)
            for vp in views:
                with open(os.path.join(sample_root, vp, fname.format(vp)), 'rb') as f:
                    data = pickle.load(f)
                f.close()
                name = '{}_{}_{}'.format(target, item, vp)
                if data.shape[1:] == shape:
                    if target in train_set:
                        np.save(os.path.join(dst, 'train', name), data)
                    elif target in test_set:
                        np.save(os.path.join(dst, 'test', name), data)
                    else:
                        unused.append(name)
                else:
                    print('Warning: Irregular sample at {}'.format(name))
                    unused.append(name)
        t_remain = (time.time()-st)/(i+1) * (target_num-i-1)
        print('Target {} completed, remain time: {}h-{}m-{}s'.format(target, int(t_remain//3600), int(t_remain%3600//60), int(t_remain%60)))
    print('Unused samples: ', unused)

if __name__ == '__main__':
    if lidargait:
        data_root = 'SUSTech1K-Released-pkl'
        dst = 'SUSTech1K-syncsilh'
    else:
        dst = data_root+'/tmp'

    if os.path.exists(dst):
        os.system('rm -rf {}'.format(dst))
    os.system('mkdir {}'.format(dst))
    os.system('mkdir {}/train'.format(dst))
    os.system('mkdir {}/test'.format(dst))

    with open('SUSTech1K.json', 'rb') as f:
        partition = json.load(f)
    f.close()

    if lidargait:
        create_raw(data_root, dst, fname, partition)
    else:
        move_data(data_root, dst, partition)

    #inference
    #data_root = '/home/sx-zhang/SUSTech1K/SUSTech1K-Released-voxel.20'
    #os.system('mkdir /home/sx-zhang/SUSTech1K/inference/dev_0')
    #create_inference(data_root, 0, partition['TEST_SET'])
    #for noise in [0.005,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085,0.095]:
    #    print('create data set with noise {}'.format(noise))
    #    data_root = '/home/sx-zhang/SUSTech1K/SUSTech1K-Released-voxel.20dev{}'.format(noise)
    #    if os.path.exists('/home/sx-zhang/SUSTech1K/inference/dev_{}'.format(noise)):
    #        os.system('rm -rf /home/sx-zhang/SUSTech1K/inference/dev_{}'.format(noise))
    #    os.system('mkdir /home/sx-zhang/SUSTech1K/inference/dev_{}'.format(noise))
    #    create_inference(data_root, noise, partition['TEST_SET'])
