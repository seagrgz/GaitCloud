#!/usr/bin/env python

import numpy as np
import rclpy
import pickle
import os
import time
import argparse
import struct
import matplotlib.pyplot as plt
from sensor_msgs_py import point_cloud2
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField

import sys
sys.path.append('/home/work/sx-zhang/GaitCloud-master')
sys.path.append('/home/work/sx-zhang/GaitCloud-master/util')
from pretreatment import seq_voxelize
from models.module import PCA_image

#sample = 371
disp_freq = 3
display_fullscene = True
trace = True
box_size = [1.25,1.25,2]
res = [0.03125,0.03125,0.03125]
dilution = 1
noise = 0
tfusion = 'tmp'
tmp_diff = False
compress = True

def get_parser():
    parser = argparse.ArgumentParser(description='Path to numpy array')
    parser.add_argument('-p', '--path', default=None, type=str)
    args = parser.parse_args()
    return args.path

class VisualArray(Node):
    def __init__(self):
        super().__init__('visual_array')
        self.pub = self.create_publisher(PointCloud2, 'v_array', 10)
        self.field = [PointField(name='x', offset=0, datatype=7, count=1),
                PointField(name='y', offset=4, datatype=7, count=1),
                PointField(name='z', offset=8, datatype=7, count=1),
                PointField(name='rgb', offset=12, datatype=6, count=1)]
        self.points = []
        self.header = Header(frame_id = 'velodyne')

    def publish(self, message=None):
        pubdata = point_cloud2.create_cloud(self.header, self.field, self.points)
        self.pub.publish(pubdata)
        print('Publishing ', message)

class FrameChecker(Node):
    def __init__(self, display_fullscene):
        super().__init__('frame_checker')
        self.pub = self.create_publisher(PointCloud2, 'frame_xyz', 10)
        if display_fullscene:
            self.field = [PointField(name = 'x', offset = 0, datatype = 7, count = 1),
                            PointField(name = 'y', offset = 4, datatype = 7, count = 1),
                            PointField(name = 'z', offset = 8, datatype = 7, count = 1)]
        else:
            self.field = [PointField(name = 'x', offset = 0, datatype = 7, count = 1),
                            PointField(name = 'y', offset = 4, datatype = 7, count = 1),
                            PointField(name = 'z', offset = 8, datatype = 7, count = 1),
                            PointField(name = 'intensity', offset = 12, datatype = 2, count = 1)]
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
    checker = FrameChecker(display_fullscene)
    data = load_raw(data_root)
    if display_fullscene:
        sample_buff = []
        for iframe in range(len(data)):
            if iframe%disp_freq == 0:
                points = data[iframe]
                #points[:,1] -= iframe/2
                if trace:
                    checker.points += points.tolist()
                else:
                    checker.points = points.tolist()
                checker.publish(iframe)
                time.sleep(0.1)
        if sample_buff != []:
            print('Sample repaired')
            save_repaired(sample_buff, data_root)
        else:
            print('Sample plot complete')
    else:
        ch_cal = np.load('elevation_128.npy')
        voxel_sample = []
        sample_len = len(data)
        voxel_frame = seq_voxelize(data, 'test', box_size=box_size, res=res, dilution=dilution, noise=noise, compress=compress)
        voxel_frame = np.transpose(voxel_frame, (0,2,3,1))
        fnum = len(voxel_sample)
        if (fnum > 1) and tmp_diff:
            voxel_frame = temporal_differ(np.stack(voxel_sample[fnum-2:]), reduction='sum')
            voxel_sample[fnum-2] = voxel_frame
        points_4d = np.nonzero(voxel_frame)
        points = np.transpose(points_4d[1:])
        points[:,0] = points[:,0]+points_4d[0]*0.5*(box_size[0]**2)/(res[0]*disp_scale)
        checker.points = points
        
        for i in range(5):
            time.sleep(0.05)
            checker.publish()
            time.sleep(0.05)

        #if not trace:
        #    if tmp_diff:
        #        voxel_sample = np.delete(voxel_sample, -1, axis=0)
        #    sample_final = np.asarray(voxel_sample) #Use entire sequence
        #    if compress:
        #        sample_final = dim_compress(sample_final) #(T, z, x, y)
        #    sample_final = np.transpose(sample_final, (0,2,3,1))
        #    if sample_final.min() < 0:
        #        sample_final[np.nonzero(sample_final)] += 20 
        #    if 'tmp' in tfusion:
        #        offset = 0
        #        for f in range(sample_final.shape[0]):
        #            points = np.transpose(np.nonzero(sample_final[f]))/disp_scale
        #            points[:,0] = points[:,0]+offset
        #            checker.points += np.append(points, sample_final[f][np.nonzero(sample_final[f])][:,np.newaxis]+20, axis=1).tolist()
        #            offset += 0.5*(box_size[0]**2)/(res[0]*disp_scale)
        #    else:
        #        sample_final = np.count_nonzero(sample_final, axis=0) #[z, x, y]
        #        points = np.transpose(np.nonzero(sample_final))/disp_scale
        #        checker.points = np.append(points, sample_final[np.nonzero(sample_final)][:,np.newaxis], axis=1).tolist()
        #    for i in range(5):
        #        checker.publish()

#train sample located at SUSTech1K-Released-voxel
def display_train(data_root):
    disp_scale = 80
    checker = FrameChecker(False)
    _data = np.load(data_root)
    if isinstance(_data, np.lib.npyio.NpzFile):
        _data = _data['addons']
        #_data = _data['main']
    if len(_data.shape) == 4:
        _data = np.concatenate([_data[i,...] for i in range(len(_data))], axis=-1) #y compressed array
        #_data = _data[3]
    elif len(_data.shape) == 3:
        pass
    else:
        raise NotImplementedError('Array with {} dimension(s) is not supported'.format(len(_data.shape)))
    if _data.shape[0] == 3:
        data = _data.transpose(1,2,0)
        coors = np.transpose(np.nonzero(data.sum(axis=-1)))
        for p in coors:
            color = data[tuple(p)]
            p[1] += (p[1]%2)*2
            p = p/disp_scale
            p = [p[0], p[1], p[2]]
            #rgb = struct.unpack('I', struct.pack('BBBB', *color.astype('uint'), 0))
            checker.points.append(p+list([color.sum().astype('float32')]))
    else:
        data = np.transpose(_data, (1,2,0))
        #np.set_printoptions(threshold=np.inf)
        points = np.transpose(np.nonzero(data))
        #points[:,0] += ((points[:,1]//2)*data.shape[1]/2).astype('int64')
        #points[:,1] = points[:,1]%2
        points[:,1] += points[:,1]//2
        checker.points = np.append(points/disp_scale, data[np.nonzero(data)][:,np.newaxis], axis=1).tolist()

    for i in range(5):
        checker.publish(data_root)
        time.sleep(0.25)
        checker.publish(data_root)
        time.sleep(0.25)

#intermidiate results
def display_array(data_root):
    save_name = data_root.split('/')[-2:]
    save_name = save_name[0] + save_name[1][:-4]
    checker = VisualArray()
    _data = np.squeeze(np.load(data_root))
    disp_scale = 80
    p_buff = []
    if len(_data.shape) == 4 and 'attention' not in data_root:
        data = np.transpose(PCA_image(_data), (1,2,0,3)) #(z, x, y, c) -> (x, y, z, c)
        x, y, z, _ = data.shape
        coors = np.transpose(np.nonzero(data.sum(axis=-1)))
        for p in coors:
            color = data[tuple(p)]
            rgb = struct.unpack('I', struct.pack('BBBB', *color.astype('uint'), 0))
            p_buff.append(list(p)+list(rgb)+list(color))
    elif len(_data.shape) == 3:
        data = np.transpose(_data, (1,2,0))
        x, y, z = data.shape
        np.set_printoptions(threshold=np.inf)
        coors = np.transpose(np.nonzero(data))
        p_buff = np.append(coors, data[np.nonzero(data)][:,np.newaxis]*255, axis=1)
    elif len(_data.shape) == 4:
        data = np.transpose(_data, (1,2,3,0))
        data = PCA_image(data) #(c, x, y, z) -> (x, y, z, c)
        x, y, z, _ = data.shape
        coors = np.transpose(np.nonzero(data.sum(axis=-1)))
        for p in coors:
            color = data[tuple(p)]
            rgb = struct.unpack('I', struct.pack('BBBB', *color.astype('uint'), 0))
            p_buff.append(list(p)+list(rgb)+list(color))
    else:
        raise NotImplementedError('Array with {} dimension(s) is not supported'.format(len(_data.shape)))

    dim = 1
    if not isinstance(p_buff, np.ndarray):
        p_buff = np.asarray(p_buff) #[x, y, z, rgb, (r, g, b)]
    #color_range = (0,max(p_buff[:,3]))
    for s in range(data.shape[dim]):
        s_points = p_buff[p_buff[:,dim] == s]
        if s_points.shape[1] == 7:
            img_p = np.delete(s_points, 3, axis=1).copy()
        else:
            img_p = s_points.copy()
        plt_p = s_points[:,:4].copy()

        #expand along y
        #plt_p[:,0] += (x+2)*s
        #plt_p[:,1] = 0

        plt_p[:,:3] = plt_p[:,:3]/disp_scale
        checker.points += plt_p.tolist()
        save_as_img(img_p, img_size=(x,z), savename='visual_results/{}{}.png'.format(save_name,s))
        checker.publish(s)
        time.sleep(0.1)
        #checker.points.clear()

#def bg_clear(points, bg, sh):
#    """
#    points: list[x,y,z,c]
#    """
#    new_points = []
#    for p in points:
#        if bg-sh < points < bg+sh:
            

def save_as_img(points, v_lims=None, img_size=(10,16), scale=1, savename='test_out.png'):
    dpi = 100               # Image resolution
    pspace = int(400/img_size[0])
    if points.shape[1] == 6:
        colors = points[:,3:]/255
        cmap = None
    else:
        colors = points[:,3]
        cmap = 'jet'
    x_max, y_max = img_size[0]*pspace, img_size[1]*pspace
    points[:,:3] = (points[:,:3]+0.5)*scale*pspace #adjust pixel coordinates
    fig, ax = plt.subplots(figsize=(x_max/dpi, y_max/dpi), dpi=dpi)
    ax.scatter(points[:,0], points[:,2], s=(pspace*72/dpi)**2, c=colors, marker='s', cmap=cmap, linewidths=0, alpha=1)
    ax.set_facecolor((0, 0, 0)) # Set regions with no points to black
    ax.axis('scaled')              # {equal, scaled}
    ax.xaxis.set_visible(False)    # Do not draw axis tick marks
    ax.yaxis.set_visible(False)    # Do not draw axis tick marks
    plt.xlim([0, x_max])   # prevent drawing empty space outside of horizontal FOV
    plt.ylim([0, y_max])   # prevent drawing empty space outside of vertical FOV
    plt.tight_layout(pad=0)
    fig.savefig(savename, dpi=dpi)
    plt.close()

if __name__ == '__main__':
    rclpy.init(args=None)
    data_root = get_parser()
    if 'SUSTech1K' not in data_root:
        display_array(data_root)
    elif '.npz' in data_root:
        display_train(data_root)
    elif '.pkl' in data_root:
        display_raw(data_root)

    #if display_fullscene:
    #    data_root = data_source+'SUSTech1K-Released-pkl/0747/01-ub/270-far/00-270-far-LiDAR-PCDs.pkl'
    #    display_raw(data_root)
    #else:
    #    view = '270-far'
    #    split = '01-ub' #'00-nm', '01-cr', '01-bg', '01-ub', '01-nm', '01-oc', '01-cl'
    #    sample_root = data_source+'SUSTech1K-Released-voxel.20/tmp/probe'
    #    data_root = data_source+'SUSTech1K-Released-pkl/{}/{}/{}/00-{}-LiDAR-PCDs.pkl'
    #    sample_list = os.listdir(sample_root)
    #    direction = []
    #    for sample in sample_list:
    #        if '_{}_'.format(view) in sample:
    #            display_train(os.path.join(sample_root, sample))
    #            #if split in sample:
    #            #    print('plotting: ', sample)
    #            #    true_view = display_raw(data_root.format(sample[-10:-6], split, view, view))
    #            #    direction.append([true_view,sample])
    #    for item in direction:
    #        print(item)
