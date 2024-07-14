#!/usr/bin/env python

import numpy as np
import pickle
import random
import math
import yaml
import argparse
import os
import shutil
import sys
import time
from voxelize import frame_voxelize, frame_rotation

#dataset setup
tfusion = False
w = 40
l = 40
h = 64

def get_parser():
    parser = argparse.ArgumentParser(description='Specify split and viewpoint to run point cloud voxelization')
    parser.add_argument('-s','--split',  type=str)
    parser.add_argument('-v','--viewpoint',  type=str)
    args = parser.parse_args()
    return args.split, args.viewpoint


#def create_GaitDatabase():
#    global w, h, l, frame_num
#    #set information
#    targetlist = ['selftest', 'tagata', 'hirai', 'hara', 'murase', 'matsuoka']
#    numlist = [100, 100, 100, 100, 100, 100]
#
#    #data path
#    src = '/home/sx-zhang/gait_database/{}/sample{}/0{}_{}.npy'
#    dst = '/home/sx-zhang/gait_database/dataset_voxel/dir{}/{}.npy'
#    dst_combine = '/home/sx-zhang/gait_database/dataset_voxel/combined/{}.npy'
#
#    #load error list
#    with open('/home/sx-zhang/gait_database/check_log/error_list_{}.pkl'.format(frame_num), 'rb') as fp:
#        errorlist = pickle.load(fp)
#        print('error list for {} frame load complete'.format(frame_num))
#    print(errorlist)
#    count = 0
#    
#    if os.path.exists('/home/sx-zhang/gait_database/dataset_voxel/dir1'):
#        for DIR in range(1,5):
#            shutil.rmtree('/home/sx-zhang/gait_database/dataset_voxel/dir{}'.format(DIR))
#        shutil.rmtree('/home/sx-zhang/gait_database/dataset_voxel/combined')
#        print('old data removed')
#
#    #save four directions separately
#    for label, target in enumerate(targetlist):
#        for samp in range(numlist[label]):
#            if samp not in errorlist[target]:
#                #sample_st = time.time()
#                for DIR in range(1,5):
#                    if not os.path.exists('/home/sx-zhang/gait_database/dataset_voxel/dir{}'.format(DIR)):
#                        os.makedirs('/home/sx-zhang/gait_database/dataset_voxel/dir{}'.format(DIR))
#                    data = np.load(src.format(target, samp, DIR, frame_num))
#                    #voxel_sample = np.zeros((w*l*h, 4)) #use coor
#                    voxel_sample = np.zeros((w, l, h))
#                    for iframe in range(int(frame_num)):
#                        voxel_frame = frame_voxelize(data[iframe])[...,-1]
#                        shape = voxel_frame.shape
#                        #print(shape)
#                        #fusion_st = time.time()
#                        voxel_sample += voxel_frame
#                        #for index in np.nonzero(voxel_flat[:,3])[0]:
#                        #    if voxel_sample[index,3] == 0:
#                        #        voxel_sample[index] += voxel_flat[index]
#                        #    else:
#                        #        voxel_sample[index,3] += 1
#                        #fusion_ed = time.time()
#                    #print(data[-1,0,0,0])
#                    labeled_sample = np.array([voxel_sample.astype('int8'),data[-1,0,0,0].astype('int8')], dtype=object)
#                    #save_st = time.time()
#                    np.save(dst.format(DIR, str(count).zfill(6)), labeled_sample, allow_pickle=True) #one sample saved
#
#                    #save combined set
#                    if not os.path.exists('/home/sx-zhang/gait_database/dataset_voxel/combined'):
#                        os.makedirs('/home/sx-zhang/gait_database/dataset_voxel/combined')
#                    np.save(dst_combine.format(str(count*4+DIR-1).zfill(6)), labeled_sample, allow_pickle=True) #one sample saved
#                    #save_ed = time.time()
#                    #print('fusion time:', fusion_ed-fusion_st, 'save time:', save_ed-save_st)
#
#                #sample_ed = time.time()
#                count += 1
#                print('{} sample{} moved'.format(target, samp))
#                #print('sample time:', sample_ed-sample_st)

def get_dir(data, st, ed, view):
    err = np.deg2rad(10)
    eps = 0.001
    dir_all = []
    for f in range(st,ed):
        f_n = min(f+2, ed)
        c0_x = np.average(data[f][:,0])
        c0_y = np.average(data[f][:,1])
        cn_x = np.average(data[f_n][:,0])
        cn_y = np.average(data[f_n][:,1])
        delta_x = cn_x-c0_x
        if delta_x == 0:
            delta_x += eps
        a = np.arctan((cn_y-c0_y)/(delta_x))
        if cn_x-c0_x < 0:
            if '90' in view:
                a = a - np.pi
            else:
                a = a + np.pi
        dir_all.append(a)
    if len(dir_all) > 1:
        mid = sorted(dir_all)[int(len(dir_all)/2)]
        filtered = [a for a in dir_all if abs(a-mid) < err]
        alpha = sum(filtered)/len(filtered)
    else:
        alpha = np.random.uniform(0, 2*np.pi)
    print(alpha)
    #c0_x = np.average(data[f][:,0])
    #c0_y = np.average(data[f][:,1])
    #cn_x = np.average(data[f_n][:,0])
    #cn_y = np.average(data[f_n][:,1])
    #alpha = np.arctan((cn_y-c0_y)/(cn_x-c0_x))
    #if cn_x-c0_x < 0:
    #    alpha += np.pi
    return -alpha

def random_head_sample(seq, frame_num):
    '''
    input   : [T, ...]
    output  : [S, ...]
    '''
    #frame_num = max(frame_num + np.random.randint(-3,3), 1)
    sampled_frames = []
    seq_len = len(seq)
    indices = list(range(seq_len))
    if seq_len < frame_num:
        it = math.ceil(frame_num/seq_len)
        seq_len = seq_len * it
        indices = indices * it
    start = random.choice(list(range(0, seq_len - frame_num + 1)))
    end = start + frame_num
    idx_lst = list(range(seq_len))
    idx_lst = idx_lst[start:end]
    assert len(idx_lst) == frame_num
    indices = [indices[i] for i in idx_lst]
    for i in indices:
        sampled_frames.append(seq[i])
    sampled_frames = np.stack(sampled_frames)
    return sampled_frames

def create_SUSTech(data_root, frame_num, split=None, viewpoint=None):
    global w, h, l, tfusion
    if split==None and viewpoint==None:
        split, viewpoint = get_parser()
    src = '/home/sx-zhang/SUSTech1K/SUSTech1K-Released-pkl'
    dst = data_root
    sample_list = sorted(os.listdir(src)) 
    sample_num = len(sample_list)
    enable_list = {}
    enable_start = 0
    #sample_point_count = {}
    state = 0
    for sample in sample_list:
        sample_root = os.path.join(src, sample)
        attributes = list(os.listdir(sample_root))
        #print(attributes)
        if len(attributes) > 0:
            for item in attributes:
                namespace = '_'.join([item, viewpoint, sample]) + '.npy'
                save_path = os.path.join(dst, viewpoint, namespace)
                if split in item and not os.path.exists(save_path):
                    data_root = os.path.join(sample_root, '{}/{}/00-{}-LiDAR-PCDs.pkl'.format(item, viewpoint, viewpoint))
                    if os.path.exists(data_root):
                        with open(data_root, 'rb') as f:
                            data = pickle.load(f)
                        f.close()
                        sample_len = len(data)
                        sample_depth = []
                        #(z, x, y)
                        voxel_sample = []
                        #sample_point_count['sample_{}'.format(sample)] = []
                        #extract continous and valid frames from raw data
                        for iframe in range(len(data)):
                            data_frame = data[iframe]
                            try:
                                if len(data_frame) > 0:
                                    if state == 0:
                                        enable_start = iframe
                                        #alpha = get_dir(data, iframe, min(iframe+20, sample_len-1), viewpoint)
                                        state = 1
                                    #data_frame = frame_rotation(data_frame, alpha)
                                    voxel_frame, frame_depth = frame_voxelize(data_frame)
                                    voxel_frame[...,-1] = np.sqrt(np.sum(voxel_frame[...,:3]**2, axis=-1))
                                    voxel_sample.append(voxel_frame[...,-1])
                                    sample_depth.append(frame_depth)
                                    #sample_point_count['sample_{}'.format(sample)].append(point_count)
                            except IndexError as e:
                                print('{}/{}: Sample {} frame {} IndexError: {}'.format(split, viewpoint, sample, iframe, e))
                        state = 0
                        if len(voxel_sample) == 0:
                            #del sample_point_count['sample_{}'.format(sample)]
                            print('{}/{}: Sample {} discarded'.format(split, viewpoint, sample))
                        else:
                            #sampling frames from voxelized data
                            voxel_sample = np.stack(voxel_sample) #(T, h, w, l)
                            sampled_frames = random_head_sample(voxel_sample, frame_num)

                            if tfusion:
                                voxel_sample = np.asarray([sampled_frames.astype(np.float16), centroid], dtype=object)
                                np.save(save_path, voxel_sample, allow_pickle=True) #one sample saved
                            else:
                                voxel_stack_sample = np.count_nonzero(sampled_frames, axis=0)
                                centroid = sum(sample_depth)/len(sample_depth)
                                voxel_stack_sample = np.asarray([voxel_stack_sample.astype('int8'), centroid], dtype=object)
                                np.save(save_path, voxel_stack_sample, allow_pickle=True) #one sample saved
                            print('{}/{}: Sample {} completed'.format(viewpoint, split, sample))
                            if sample not in enable_list.keys():
                                enable_list[sample] = enable_start
                    else:
                        print('FileNotFound: {}'.format(data_root))
        else:
            print('No data in {}'.format(sample))
    #with open(os.path.join(dst, viewpoint, 'sample_list.yaml'), 'w') as f:
    #    yaml.dump(enable_list, f, allow_unicode=True)
    #f.close()
    print('{} sample used'.format(len(enable_list)))
    print('{} sample discarded'.format(sample_num-len(enable_list)))
    #return sample_point_count

def find_minimum_cluster(point_counter):
    min_count = 9999
    min_pos = 'None'
    for key, value in point_counter.items():
        local_min = min(value)
        if local_min < min_count:
            min_count = local_min
            min_pos = '{} frame {}'.format(key, value.index(local_min))
    return min_count, min_pos

if __name__ == '__main__':
    point_counter = create_SUSTech('/home/sx-zhang/SUSTech1K/SUSTech1K-Released-voxel', 20)
