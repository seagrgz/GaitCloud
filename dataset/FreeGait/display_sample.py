#!/usr/bin/env python

import numpy as np
import rclpy
import pickle
import os
import time
import argparse
import struct
from sensor_msgs_py import point_cloud2
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField

import sys
sys.path.append('/home/work/sx-zhang/GaitCloud-master')
sys.path.append('/home/work/sx-zhang/GaitCloud-master/util')
from pretreatment import frame_voxelize, frame_rotation, assign_ID, frame_dilution, add_noise, frame_clean, get_frame_dir, temporal_differ, dim_compress
from models.module import PCA_image

#sample = 371
display_fullscene = True
trace = True
ch_cali = [100, -0.01]
box_size = [1.25,1.25,2]
res = [0.03125,0.03125,0.03125]
dilution = 1
noise = 0
tfusion = 'tmp'
tmp_diff = False
compress = True

def get_parser():
    parser = argparse.ArgumentParser(description='Path to numpy array')
    parser.add_argument('-p', '--path', type=str)
    args = parser.parse_args()
    return args.path

class VisualArray(Node):
    def __init__(self):
        super().__init__('visual_array')
        self.pub = self.create_publisher(PointCloud2, 'v_array', 10)
        self.field = [PointField(name='x', offset=0, datatype=7, count=1),
                PointField(name='y', offset=4, datatype=7, count=1),
                PointField(name='z', offset=8, datatype=7, count=1),
                PointField(name='rgb', offset=12, datatype=7, count=1)]
        self.points = []
        self.header = Header(frame_id = 'velodyne')

    def publish(self, message=None):
        pubdata = point_cloud2.create_cloud(self.header, self.field, self.points)
        self.pub.publish(pubdata)
        print('Publishing ', message)

class FrameChecker(Node):
    def __init__(self, field_l=3):
        super().__init__('frame_checker')
        self.pub = self.create_publisher(PointCloud2, 'frame_xyz', 10)
        if field_l == 3:
            self.field = [PointField(name = 'x', offset = 0, datatype = 7, count = 1),
                            PointField(name = 'y', offset = 4, datatype = 7, count = 1),
                            PointField(name = 'z', offset = 8, datatype = 7, count = 1)]
        elif field_l == 4:
            self.field = [PointField(name = 'x', offset = 0, datatype = 7, count = 1),
                            PointField(name = 'y', offset = 4, datatype = 7, count = 1),
                            PointField(name = 'z', offset = 8, datatype = 7, count = 1),
                            PointField(name = 'intensity', offset = 12, datatype = 2, count = 1)]
        else:
            raise NotImplementedError('Not supported field num! (Expect 3/4 but got {})'.format(field_l))
        self.points = []
        self.header = Header(frame_id = 'velodyne')

    def publish(self, name=None):
        pubdata = point_cloud2.create_cloud(self.header, self.field, self.points)
        self.pub.publish(pubdata)
        print('publishing ', name, len(self.points))

def load_raw(data_root):
    with open(data_root, 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data

def save_repaired(data, data_root):    
    with open(data_root, 'wb') as f:
        pickle.dump(data, f)
    f.close()
                       
#raw sample located     at SUSTech1K-Released-pkl
def display_raw(data_root):
    global frame_num, display_fullscene, trace, dilution, noise, tmp_diff, box_size, res
    disp_scale = 80
    data = load_raw(data_root)
    if display_fullscene:
        checker = FrameChecker(len(data[0][0]))
        sample_buff = []
        for iframe in range(len(data)):
            if iframe < 9999:
                points = data[iframe].tolist()
                if trace:
                    checker.points += points
                else:
                    checker.points = points
                checker.publish(iframe)
                time.sleep(0.1)
        if sample_buff != []:
            print('Sample repaired')
            save_repaired(sample_buff, data_root)
        else:
            print('Sample plot complete')
    else:
        checker = FrameChecker(4)
        ch_cal = np.load('elevation_128.npy')
        voxel_sample = []
        sample_len = len(data)
        offset = 0
        data = [frame[:,:3] for frame in data if len(frame)>0]
        for iframe in range(len(data)):
            if iframe == 0:
                ground = min([min(frame[:,2]) for frame in data[iframe:]])
            if dilution > 1:
                data_frame, Id = assign_ID(data[iframe], ch_cal)
                data_frame, Id = frame_dilution(data_frame, Id, stride=dilution)
            else:
                data_frame = np.asarray(data[iframe])
            if noise > 0:
                data_frame = add_noise(data_frame, noise)
            alpha = get_frame_dir(data[max(iframe-1,0):min(iframe+2,len(data))])
            data_frame = frame_rotation(data_frame, alpha) #Rotation
            voxel_frame, frame_depth = frame_voxelize(data_frame, ground, box_size, res) #Voxelization
            #voxel_frame = np.sqrt(np.sum(voxel_frame**2, axis=-1)) #depth
            voxel_frame = np.where((voxel_frame!=0).any(axis=-1), 1, 0) #PH
            voxel_sample.append(voxel_frame)
            voxel_frame = np.transpose(voxel_frame, (1,2,0))
            fnum = len(voxel_sample)
            if (fnum > 1) and tmp_diff:
                voxel_frame = temporal_differ(np.stack(voxel_sample[fnum-2:]), reduction='sum')
                voxel_sample[fnum-2] = voxel_frame
            points = np.transpose(np.nonzero(voxel_frame))/disp_scale
            points[:,0] = points[:,0]+offset
            if trace:
                checker.points += np.append(points, voxel_frame[np.nonzero(voxel_frame)][:,np.newaxis], axis=1).tolist()
                offset += 0.5*(box_size[0]**2)/(res[0]*disp_scale)
            else:
                continue
            time.sleep(0.05)
            checker.publish()
            time.sleep(0.05)

        if not trace:
            if tmp_diff:
                voxel_sample = np.delete(voxel_sample, -1, axis=0)
            sample_final = np.asarray(voxel_sample) #Use entire sequence
            if compress:
                sample_final = dim_compress(sample_final) #(T, z, x, y)
            sample_final = np.transpose(sample_final, (0,2,3,1))
            if sample_final.min() < 0:
                sample_final[np.nonzero(sample_final)] += 20 
            if 'tmp' in tfusion:
                offset = 0
                for f in range(sample_final.shape[0]):
                    points = np.transpose(np.nonzero(sample_final[f]))/disp_scale
                    points[:,0] = points[:,0]+offset
                    checker.points += np.append(points, sample_final[f][np.nonzero(sample_final[f])][:,np.newaxis], axis=1).tolist()
                    offset += 0.5*(box_size[0]**2)/(res[0]*disp_scale)
            else:
                sample_final = np.count_nonzero(sample_final, axis=0) #[z, x, y]
                points = np.transpose(np.nonzero(sample_final))/disp_scale
                checker.points = np.append(points, sample_final[np.nonzero(sample_final)][:,np.newaxis], axis=1).tolist()
            for i in range(5):
                checker.publish()

#train sample located at SUSTech1K-Released-voxel
def display_train(data_root):
    disp_scale = 80
    checker = FrameChecker(4)
    _data = np.load(data_root)
    if isinstance(_data, np.lib.npyio.NpzFile):
        _data = _data['main']
    if len(_data.shape) == 4:
        _data = np.transpose(_data.sum(axis=3), (1,2,0)) #y compressed array
        #_data = _data[0]
    elif len(_data.shape) == 3:
        pass
    else:
        raise NotImplementedError('Array with {} dimension(s) is not supported'.format(len(_data.shape)))
    data = np.transpose(_data, (1,2,0))
    print(data.shape)
    #np.set_printoptions(threshold=np.inf)
    points = np.transpose(np.nonzero(data))
    checker.points = np.append(points/disp_scale, data[np.nonzero(data)][:,np.newaxis], axis=1).tolist()
    checker.publish(data_root)
    time.sleep(0.25)
    checker.publish(data_root)
    time.sleep(0.25)
    checker.points = []

#intermidiate results
def display_array(data_root):
    checker = VisualArray()
    _data = np.squeeze(np.load(data_root))
    disp_scale = 80
    sh = 10
    if len(_data.shape) == 4:
        data = np.transpose(PCA_image(_data), (1,2,0,3)) #(z, x, y, c) -> (x, y, z, c)
        #bg = data[4,4,4]
        bg = [-100, -100, -100]
        print(bg)
        coors = np.transpose(np.nonzero(data.sum(axis=-1)))
        for p in coors:
            color = data[tuple(p)]
            if (abs(color[0]-bg[0]) < sh) and (abs(color[1]-bg[1]) < sh) and (abs(color[2]-bg[2]) < sh):# or p[0]%36 < 4 or p[1]%36 < 4 or p[2]%60 < 4:
                continue
            else:
                rgb = struct.unpack('I', struct.pack('BBBB', *color.astype('uint'), 0))
                checker.points.append(list(p/disp_scale)+list(rgb))
    elif len(_data.shape) == 3:
        print(_data.shape)
        data = np.transpose(_data, (1,2,0))
        coors = np.transpose(np.nonzero(data))
        print(max(data[np.nonzero(data)]))
        checker.points = np.append(coors/disp_scale, data[np.nonzero(data)][:,np.newaxis], axis=1).tolist()
    else:
        raise NotImplementedError('Array with {} dimension(s) is not supported'.format(len(_data.shape)))

    #for i in range(5):
    #    checker.publish()

    p_buff = checker.points
    checker.points = []
    dim = 1
    for s in range(data.shape[dim]):
        for point in p_buff:
            if point[dim] == s/disp_scale:
                checker.points.append(point)
        checker.publish(s)
        time.sleep(1)
        checker.points.clear()

if __name__ == '__main__':
    rclpy.init(args=None)
    data_root = 'FreeGait-voxel.20tmpspatcentcomp/0564-devid_1-seq_06_new.npz'
    display_train(data_root)

    #data_root = 'FreeGait/0930/devid_2/seq_01_new/lidar/lidar.pkl'
    #display_raw(data_root)

    #data_root = get_parser()
    #display_array(data_root)
