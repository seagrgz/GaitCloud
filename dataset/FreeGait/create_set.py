#!/usr/bin/env python

import os
import json
import yaml
import time
import pickle
import itertools
import random
import numpy as np

data_root = 'FreeGait-voxel.20tempspat20centcomp'
lidargait = False
#g_size = 1
                    
def move_data(root, dst, partition):
    train_set = partition["TRAIN_SET"]
    test_set = partition["TEST_SET"]
    probe_set = partition["PROBE_SET"]

    #use all samples
    sample_list = sorted(os.listdir(root))
    sample_num = len(sample_list) - 1
    st = time.time()
    for i, sample in enumerate(sample_list):
        if i%100==0 and i!=0:
            t_remain = (time.time()-st)/(i+1) * (sample_num-i-1)
            print('Processing...{}%, remain time: {}h-{}m-{}s'.format(round(i*100/sample_num,2), int(t_remain//3600), int(t_remain%3600//60), int(t_remain%60)))
        if '.npz' in sample:
            try:
                target, dev, seq = sample[:-4].split('-')
            except Exception as e:
                print(sample, ': ', e)
                raise RuntimeError('Unexpect filename in root folder')
        else:
            continue
        if target in train_set:
            os.system('cp {}/{} {}/train/{}'.format(root, sample, dst, sample))
        if target in test_set:
            os.system('cp {}/{} {}/test/{}'.format(root, sample, dst, sample))

def create_depthimg(data_root, dst, partition):
    train_samples = partition["TRAIN_SET"]
    test_samples = partition["TEST_SET"]
    label_list = sorted(os.listdir(data_root))
    train_set = [label for label in train_samples if label in label_list]
    test_set = [label for label in test_samples if label in label_list]

    target_num = len(label_list)
    st = time.time()
    for i, target in enumerate(label_list):
        dev_root = os.path.join(data_root, target)
        devs = os.listdir(dev_root)
        for dev in devs:
            seq_root = os.path.join(dev_root, dev)
            seq_names = os.listdir(seq_root)
            for seq in seq_names:
                sample_root = os.path.join(seq_root, seq, 'range_pkl')
                if len(os.listdir(sample_root)) > 1:
                    raise RuntimeError('Multiple samples exist in the root!')
                else:
                    sample_path = os.path.join(sample_root, 'range_pkl.pkl')

                with open(os.path.join(sample_path), 'rb') as f:
                    data = pickle.load(f)
                f.close()
                if target in train_set:
                    np.save(os.path.join(dst, 'train', '{}-{}-{}'.format(target, dev, seq)), data)
                if target in test_set:
                    np.save(os.path.join(dst, 'test', '{}-{}-{}'.format(target, dev, seq)), data)
        t_remain = (time.time()-st)/(i+1) * (target_num-i-1)
        print('Target {} completed, remain time: {}h-{}m-{}s'.format(target, int(t_remain//3600), int(t_remain%3600//60), int(t_remain%60)))

#def create_inference(root, dst, partition, g_size):
#    test_set = partition["TEST_SET"]
#    sample_list = os.listdir(root)
#    sample_num = len(sample_list)
#    OK = 0
#    st = time.time()
#    for target in test_set:
#        sample_pool = [item for item in sample_list if target in item]
#        if len(sample_pool) > 0:
#            g_ids = np.random.choice(len(sample_pool), size=(min(len(sample_pool),g_size),), replace=False)
#            for g_id in g_ids:
#                os.system('cp {}/{} {}/gallery/{}'.format(root, sample_pool[g_id], dst, sample_pool[g_id]))
#            for i in range(len(sample_pool)):
#                if i not in g_ids:
#                    os.system('cp {}/{} {}/probe/{}'.format(root, sample_pool[i], dst, sample_pool[i]))
#        else:
#            continue
#        OK += len(sample_pool)
#        t_remain = (time.time()-st)/OK * (sample_num-OK)
#        print('Target {} completed, remain time: {}h-{}m-{}s'.format(target, int(t_remain//3600), int(t_remain%3600//60), int(t_remain%60)))

if __name__ == '__main__':
    if lidargait:
        data_root = 'FreeGait'
        dst = 'FreeGait-depthimg'
    else:
        dst = data_root+'/tmp'

    with open('FreeGait_Data_Split.json', 'rb') as f:
        partition = json.load(f)
    f.close()

    if os.path.exists(dst):
        print('Removing expired data...')
        os.system('rm -rf {}'.format(dst))
    os.system('mkdir {}'.format(dst))
    os.system('mkdir {}/train'.format(dst))
    os.system('mkdir {}/test'.format(dst))

    if lidargait:
        create_depthimg(data_root, dst, partition)
    else:
        move_data(data_root, dst, partition)
