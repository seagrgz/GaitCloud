#!/usr/bin/env python

import numpy as np
import os
import sys
import math
import struct
import time
from sklearn.cluster import DBSCAN

res_x = 0.03125
res_y = 0.03125
res_z = 0.03125

width_x = 1.25
width_y = 1.25
width_z = 2

w = int(width_x/res_x)
l = int(width_y/res_y)
h = int(width_z/res_z)

def frame_rotation(frame, alpha):
    """
    input: frame [n, 3]
    output: rotated frame [n, 3]
    """
    mat_R = np.array([[np.cos(alpha), -(np.sin(alpha)), 0], 
                        [np.sin(alpha), np.cos(alpha), 0],
                        [0, 0, 1]])
    #frame_trans = np.matmul(np.expand_dims(frame, 1), mat_R).squeeze()
    frame_trans = np.zeros(frame.shape)
    for index, point in enumerate(frame):
        frame_trans[index] = np.matmul(mat_R, point)
    return frame_trans

def frame_voxelize(frame):
    """
    input: frame [n, 3]
    output: voxelized_frame [x, y, z]
    """
    global res_x, res_y, res_z, w, l, h, width_x, width_y, width_z
    counter = 1

    #clustering
    if type(frame) != 'numpy.ndarray':
        frame = np.array(frame)
    cluster = DBSCAN(eps=0.1, min_samples=4).fit(frame)
    labels = cluster.labels_
    #unique_labels, counts = np.unique(labels[labels!=-1], return_counts=True)
    #frame = frame[labels==unique_labels[np.argmax(counts)]]
    f_frame = frame[labels!=-1]
    valid_frame = f_frame[f_frame.any(axis=1)]
    if len(valid_frame) > 0:
        frame = valid_frame
    #point_num = len(frame)

    min_z = np.min(frame[:,2])
    max_z = np.max(frame[:,2])
    lim_z = max_z-min_z
    if len(frame[(frame[:,2]-min_z>lim_z*0.5)&(frame[:,2]-min_z<lim_z*0.7)]) > 0:
        center_x = np.average(frame[:,0][(frame[:,2]-min_z>lim_z*0.5)&(frame[:,2]-min_z<lim_z*0.7)])
        center_y = np.average(frame[:,1][(frame[:,2]-min_z>lim_z*0.5)&(frame[:,2]-min_z<lim_z*0.7)])
    elif len(frame[:,0][(frame[:,2]-min_z>lim_z*0.7)]) > 0:
        center_x = np.average(frame[:,0][(frame[:,2]-min_z>lim_z*0.7)])
        center_y = np.average(frame[:,1][(frame[:,2]-min_z>lim_z*0.7)])
    else:
        center_x = np.average(frame[:,0])
        center_y = np.average(frame[:,1])
    frame_depth = math.sqrt(center_x**2+center_y**2)
    center_z = min_z
    frame_vox = np.zeros((h, w, l, 4))

    coor_x = ((frame[:,0]-center_x)//res_x+w/2-1).astype(int)
    coor_y = ((frame[:,1]-center_y)//res_y+l/2-1).astype(int)
    coor_z = ((frame[:,2]-center_z)//res_z).astype(int)
    bound_mask = (0 <= coor_x)&(coor_x < w)&(0 <= coor_y)&(coor_y < l)&(0 <= coor_z)&(coor_z < h)
    masked_x = coor_x[bound_mask]
    masked_y = coor_y[bound_mask]
    masked_z = coor_z[bound_mask]
    masked_frame = frame[bound_mask]
    frame_vox[masked_z, masked_x, masked_y] = np.column_stack((masked_frame[:,2], masked_frame[:,0], masked_frame[:,1], np.full(masked_frame.shape[0], counter)))
    #for i, point in enumerate(frame):
    #    if point.any() != 0:
    #        coor_x = int(((point[0]-center_x)//res_x)+w/2-1)
    #        coor_y = int(((point[1]-center_y)//res_y)+l/2-1)
    #        coor_z = int((point[2]-center_z)//res_z)

    #        if (coor_x>=0) and (coor_y>=0) and (coor_z>=0):
    #            try:
    #                frame_vox[coor_z, coor_x, coor_y] = [point[2], point[0], point[1], counter]#(labels[labels!=-1][i]+1)*256/(unique_labels[-1]+1)]
    #            except IndexError:
    #                pass
    return frame_vox, frame_depth#point_num#, np.delete(frame_flat, -1, -1) #min_xv, max_xv, min_yv, max_yv, min_zv, max_zv 

if __name__ == '__main__':
    rclpy.init(args=None)
    checker = FrameChecker()

    frame_num = 10
    DIR = 'dir1'
    data_root = '/home/sx-zhang/gait_database/dataset{}/{}'.format(frame_num, DIR)

    data_list = sorted(os.listdir(data_root))
    data_list = [item[:-4] for item in data_list]

    #board = []
    for item in data_list:
        data_path = os.path.join(data_root, item + '.npy')
        data = np.load(data_path)
        voxel_sample = np.zeros((w*l*h, 4))
        for iframe in range(int(frame_num)):
            voxel_data = frame_voxelize(data[iframe])
            voxel_flat = voxel_data.reshape(-1, voxel_data.shape[-1])/100
            for index in range(len(voxel_flat)):
                if voxel_flat[index,3] != 0:
                    if voxel_sample[index,3] == 0:
                        voxel_sample[index] += voxel_flat[index]
                    else:
                        voxel_sample[index,3] += 1

            #checker.points = voxel_flat[np.nonzero(voxel_flat[:,0])].astype('float32').tolist()
            #checker.points.extend(frame[np.nonzero(frame[:,0])])

        checker.points = voxel_sample[np.nonzero(voxel_sample[:,0])].astype('float32').tolist()
        time.sleep(2)
        checker.publish(iframe)
        time.sleep(2)
        checker.points.clear()
            #board.append(voxelize(data[iframe]))
    #board = np.array(board)
    #edgex_min = np.min(board, axis=0)[0]
    #edgey_min = np.min(board, axis=0)[2]
    #edgez_min = np.min(board, axis=0)[4]
    #edgex_max = np.max(board, axis=0)[1]
    #edgey_max = np.max(board, axis=0)[3]
    #edgez_max = np.max(board, axis=0)[5]
    #print(edgex_min, edgex_max, edgey_min, edgey_max, edgez_min, edgez_max)
