#!/usr/bin/env python

import numpy as np
import torch
import cv2
import os
import io
import sys
import math
import random
import struct
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from PIL import Image
from sklearn.cluster import DBSCAN

def seq_voxelize(
        raw_data,
        dst,
        namespace='test_seq',
        frame_num=999,
        box_size=(1.25,1.25,2),
        res=0.03125,
        dilution=1,
        noise=0,
        counter=1,
        compress=False,
        tfusion='spat',
        tmp_diff=False
        ):
    ch_cal = np.load('elevation_128.npy') #LiDAR elevation data
    save_path = os.path.join(dst, namespace)
    discard_flag = 0
    sample_depth = []
    voxel_sample = []
    data = [frame[:,:3] for frame in raw_data if len(frame)>0]
    void_ids = [i for i, frame in enumerate(raw_data) if len(frame)==0]
    for iframe in range(len(data)):
        if iframe == 0:
            ground = min([min(frame[:,2]) for frame in data[iframe:]])
        if dilution > 1:
            data_frame, Id = assign_ID(data[iframe], ch_cal)
            data_frame, Id = frame_dilution(data_frame, Id, stride=dilution)
        else:
            data_frame = np.asarray(data[iframe])
        if len(data_frame) > 0:
            if noise > 0:
                data_frame = add_noise(data_frame, noise)
            if tfusion == 'depth':
                voxel_frame = frame_to_depthimg(data_frame)
                frame_depth = 0
            else:
                alpha = get_frame_dir(data[max(iframe-1,0):min(iframe+2,len(data))])
                data_frame = frame_rotation(data_frame, alpha) #Rotation
                voxel_frame, frame_depth = frame_voxelize(data_frame, ground, box_size, res) #Voxelization
                #voxel_frame = np.sqrt(np.sum(voxel_frame**2, axis=-1)) #depth
                voxel_frame = np.where((voxel_frame!=0).any(axis=-1), 1, 0) #PH
        else:
            if tfusion == 'depth':
                voxel_frame = np.zeros((3,64,64))
                frame_depth = 0
            else:
                voxel_frame, frame_depth = frame_voxelize(data_frame, ground, box_size, res) #Voxelization
        voxel_sample.append(voxel_frame)
        sample_depth.append(frame_depth)
    if len(void_ids) > 0:
        voxel_sample = np.insert(voxel_sample, void_ids, np.zeros_like(voxel_frame), axis=0)
    if len(voxel_sample) <= 1:
        discard_flag = 1
    else:
        if tfusion == 'depth':
            np.save(save_path, np.asarray(voxel_sample, dtype='uint8')) #one sample saved
        else:
            #sampling frames from voxelized data
            if frame_num == 999:
                sampled_frames = np.asarray(voxel_sample) #Use entire sequence
            else:
                sampled_frames = random_head_sample(np.stack(voxel_sample), frame_num) #(T, h, w, l)
            centroid = sum(sample_depth)/len(sample_depth)
            if compress:
                sample_final = dim_compress(sampled_frames) #(T, z, x, y)
            else:
                sample_final = sampled_frames
            if dst == 'test':
                return sample_final
            else:
                if 'tmp' in tfusion:
                    if tmp_diff:
                        sample_final = temporal_differ(sample_final)
                    if 'spat' in tfusion:
                        #sampled_frames = np.asarray(voxel_sample) #Use entire sequence
                        sample_spat = np.count_nonzero(sampled_frames[:-1], axis=0)*counter #[z, x, y]
                        np.savez_compressed(save_path, main=sample_spat.astype('int8'), addons=sample_final.astype('int8'))
                    else:
                        np.savez_compressed(save_path, main=sample_final.astype('int8'), addons=centroid) #one sample saved
                else: #spatial only (3D)
                    if tmp_diff:
                        #temporal differ
                        sample_final = temporal_differ(sampled_frames, reduction='sum')
                    else:
                        sample_final = np.count_nonzero(sampled_frames, axis=0)*counter #[z, x, y]
                    np.savez_compressed(save_path, main=sample_final.astype('int8'), addons=centroid) #one sample saved
    return discard_flag

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

def frame_voxelize(frame, ground, box_size, res):
    """
    input: frame [n, 3]
    output: voxelized_frame [x, y, z]
    """
    width_x, width_y, width_z = box_size
    res_x, res_y, res_z = res
    w, l, h = width_x/res_x, width_y/res_y, width_z/res_z
    assert w==int(w) and l==int(l) and h==int(h), 'box_size ({}) should be divisable by resolustion ({})'.format(box_size, res)
    w, l, h = int(w), int(l), int(h)

    #clustering
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    if len(frame) > 0:
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
        center_z = ground
        frame_vox = np.zeros((h, w, l, 3))

        coor_x = ((frame[:,0]-center_x)//res_x+w/2-1).astype(int)
        coor_y = ((frame[:,1]-center_y)//res_y+l/2-1).astype(int)
        coor_z = ((frame[:,2]-center_z)//res_z).astype(int)
        bound_mask = (0 <= coor_x)&(coor_x < w)&(0 <= coor_y)&(coor_y < l)&(0 <= coor_z)&(coor_z < h)
        masked_x = coor_x[bound_mask]
        masked_y = coor_y[bound_mask]
        masked_z = coor_z[bound_mask]
        masked_frame = frame[bound_mask]
        frame_vox[masked_z, masked_x, masked_y] = np.column_stack((masked_frame[:,2], masked_frame[:,0], masked_frame[:,1]))
    else:
        frame_vox = np.zeros((h, w, l))
        frame_depth = 0
    return frame_vox, frame_depth

def frame_to_depthimg(frame):
    """
    input:  ndarray[n, 3]
    output: ndarray[h, w]
    """
    img = lidar_to_2d_front_view(frame)
    return img

def lidar_to_2d_front_view(points,
                           v_res=0.2,
                           h_res=0.19188,
                           v_fov=(-25.0, 15.0),
                           val="depth",
                           cmap="jet",
                           saveto='tmp_img.png',
                           y_fudge=0.0
                           ):
    """ Takes points in 3D space from LIDAR data and projects them to a 2D
        "front view" image, and saves that image. (OpenGait: https://github.com/ShiqiYu/OpenGait/)

    Args:
        points: (np array)
            The numpy array containing the lidar points.
            The shape should be Nx4
            - Where N is the number of points, and
            - each point is specified by 4 values (x, y, z, reflectance)
        v_res: (float)
            vertical resolution of the lidar sensor used.
        h_res: (float)
            horizontal resolution of the lidar sensor used.
        v_fov: (tuple of two floats)
            (minimum_negative_angle, max_positive_angle)
        val: (str)
            What value to use to encode the points that get plotted.
            One of {"depth", "height", "reflectance"}
        cmap: (str)
            Color map to use to color code the `val` values.
            NOTE: Must be a value accepted by matplotlib's scatter function
            Examples: "jet", "gray"
        saveto: (str or None)
            If a string is provided, it saves the image as this filename.
            If None, then it just shows the image.
        y_fudge: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical range do not match the actual data.

            For a Velodyne HDL 64E, set this value to 5.
    """

    # DUMMY PROOFING
    assert len(v_fov) ==2, "v_fov must be list/tuple of length 2"
    assert v_fov[0] <= 0, "first element in v_fov must be 0 or negative"
    assert val in {"depth", "height", "reflectance"}, \
        'val must be one of {"depth", "height", "reflectance"}'


    x_lidar = - points[:, 0]
    y_lidar = - points[:, 1]
    z_lidar = points[:, 2]
    # Distance relative to origin when looked from top
    d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)
    # Absolute distance relative to origin
    # d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2, z_lidar ** 2)

    v_fov_total = -v_fov[0] + v_fov[1]

    # Convert to Radians
    v_res_rad = v_res * (np.pi/180)
    h_res_rad = h_res * (np.pi/180)

    # PROJECT INTO IMAGE COORDINATES
    x_img = np.arctan2(-y_lidar, x_lidar)/ h_res_rad
    y_img = np.arctan2(z_lidar, d_lidar)/ v_res_rad

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / h_res / 2  # Theoretical min x value based on sensor specs
    x_img -= x_min              # Shift
    x_max = 360.0 / h_res       # Theoretical max x value after shifting

    y_min = v_fov[0] / v_res    # theoretical min y value based on sensor specs
    y_img -= y_min              # Shift
    y_max = v_fov_total / v_res # Theoretical max x value after shifting

    y_max += y_fudge            # Fudge factor if the calculations based on
                                # spec sheet do not match the range of
                                # angles collected by in the data.

    # WHAT DATA TO USE TO ENCODE THE VALUE FOR EACH PIXEL
    if val == "reflectance":
        pass
    elif val == "height":
        pixel_values = z_lidar
    else:
        pixel_values = -d_lidar
        # pixel_values = 'w'

    # PLOT THE IMAGE
    cmap = "jet"            # Color map to use
    dpi = 300               # Image resolution
    fig, ax = plt.subplots(figsize=(x_max/dpi, y_max/dpi), dpi=dpi)
    ax.scatter(x_img,y_img, s=1, c=pixel_values, linewidths=0, alpha=1, cmap=cmap)
    ax.set_facecolor((0, 0, 0)) # Set regions with no points to black
    ax.axis('scaled')              # {equal, scaled}
    ax.xaxis.set_visible(False)    # Do not draw axis tick marks
    ax.yaxis.set_visible(False)    # Do not draw axis tick marks
    plt.xlim([0, x_max])   # prevent drawing empty space outside of horizontal FOV
    plt.ylim([0, y_max])   # prevent drawing empty space outside of vertical FOV
    #buf = io.BytesIO()
    fig.savefig(saveto, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    #fig.canvas.draw(dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    #w, h = fig.canvas.get_width_height()
    #buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    #buf.shape = (h, w, 4)
    #rgba = np.zeros_like(buf)
    #rgba[..., 0] = buf[..., 1]  # R
    #rgba[..., 1] = buf[..., 2]  # G
    #rgba[..., 2] = buf[..., 3]  # B
    #rgba[..., 3] = buf[..., 0]  # A
    #img = rgba[..., :3]
    plt.close()

    #buf.seek(0)
    #img = Image.open(buf)
    #img = np.array(img)[...,:3]

    img = cv2.imread(saveto)
    img = align_img(img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = img.transpose(2, 0, 1)
    data = np.asarray(data)
    return data

def align_img(img: np.ndarray, img_size: int = 64) -> np.ndarray:
    """Aligns the image to the center.
    Args:
        img (np.ndarray): Image to align.
        img_size (int, optional): Image resizing size. Defaults to 64.
    Returns:
        np.ndarray: Aligned image.
    """    
    if img.sum() <= 10000:
        y_top = 0
        y_btm = img.shape[0]
    else:
        # Get the upper and lower points
        # img.sum
        y_sum = img.sum(axis=2).sum(axis=1)
        y_top = (y_sum != 0).argmax(axis=0)
        y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)

    img = img[y_top: y_btm, :,:]

    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    ratio = img.shape[1] / img.shape[0]
    img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_CUBIC)
    
    # Get the median of the x-axis and take it as the person's x-center.
    x_csum = img.sum(axis=2).sum(axis=0).cumsum()
    x_center = img.shape[1] // 2
    for idx, csum in enumerate(x_csum):
        if csum > img.sum() / 2:
            x_center = idx
            break

    # if not x_center:
    #     logging.warning(f'{img_file} has no center.')
    #     continue

    # Get the left and right points
    half_width = img_size // 2
    left = x_center - half_width
    right = x_center + half_width
    if left <= 0 or right >= img.shape[1]:
        left += half_width
        right += half_width
        # _ = np.zeros((img.shape[0], half_width,3))
        # img = np.concatenate([_, img, _], axis=1)
    
    img = img[:, left: right,:].astype('uint8')
    return img

def assign_ID(frame, ch_cal):
    """
    assign channel id to points in one frame
    input   : [n, 3], array[128]
    output  : array[n, 3], array[n]
    """
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    hori_project = np.sqrt(np.sum(frame[...,:2]**2, axis=-1))
    beta = np.rad2deg(np.arctan(frame[...,2]/hori_project))
    abs_diff = np.abs(ch_cal[:,np.newaxis]-beta)
    Id = np.argmin(abs_diff, axis=0)
    return frame, Id

def frame_dilution(frame, Id, stride=2):
    indices = np.nonzero(Id%stride==0)[0]
    diluted = frame[indices]
    return diluted, Id[indices]

def add_noise(sample, sigma):
    assert sample.shape[-1] == 3, 'expect sample have 3 channel but got {}'.format(sample.shape[-1])
    noise_array = np.random.normal(0, sigma, sample.shape)
    noise_array[np.repeat(np.all(np.logical_not(sample), axis=-1, keepdims=True), sample.shape[-1], axis=-1)] = 0
    sample_with_noise = sample + noise_array
    return sample_with_noise

def frame_clean(frame):
    """
    remove irrelative points
    """
    if len(frame) > 0:
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        #align = np.sum(frame[:,:2]**2, axis=-1) + (-7.5)**2 - 2*frame[:,0]*(-7.5)
        #base = frame[np.argmin(align)][:2]
        base = frame[np.argmin(frame[:,1])][:2]
        dist = np.sum(frame[:,:2]**2, axis=-1) + np.sum(base**2) - 2*np.sum(frame[:,:2]*base, axis=-1)
        frame_filtered = frame[dist<2]
    else:
        frame_filtered = frame
    return frame_filtered

def get_dir(data, st, ed):
    delta_x = []
    delta_y = []
    for f in range(st,ed):
        f_n = min(f+2, ed)
        delta_x.append(np.average(data[f_n][:,0])-np.average(data[f][:,0]))
        delta_y.append(np.average(data[f_n][:,1])-np.average(data[f][:,1]))
    if len(delta_x) > 1:
        alpha = np.arctan2(sum(delta_y)/len(delta_y), sum(delta_x)/len(delta_x))
    else:
        alpha = np.random.uniform(0, 2*np.pi)
    return -alpha

def get_frame_dir(data):
    fnum = len(data)
    delta_x = []
    delta_y = []
    if not fnum in [2,3]:
        print('Warning: sequence with 2 or 3 frames is recommended, but got {}!'.format(fnum))
        if fnum < 2:
            return 0.
    for f in range(fnum-1):
        delta_x.append(np.average(data[f+1][:,0])-np.average(data[f][:,0]))
        delta_y.append(np.average(data[f+1][:,1])-np.average(data[f][:,1]))
    alpha = np.arctan2(sum(delta_y)/len(delta_y), sum(delta_x)/len(delta_x))
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

def dim_compress(seq): #record distances between outer border and centroid along y
    '''
    Compress the last dimension (default is y with input in format [T, z, x, y])
    output: [T, z, x, c]
    '''
    y_c = int(seq.shape[-1]/2)
    mask = (seq!=0)
    #range to centroid
    min_dis = np.where(mask[...,:y_c].any(axis=-1), y_c-mask.argmax(axis=-1), 0) #left (low) border
    max_dis = np.where(mask[...,y_c:].any(axis=-1), y_c-mask[...,::-1].argmax(axis=-1), 0) #right (high) border
    #min_dis = np.where(mask.any(axis=-1), y_c-mask.argmax(axis=-1), 0) #left (low) border
    #max_dis = np.where(mask.any(axis=-1), y_c-mask[...,::-1].argmax(axis=-1), 0) #right (high) border

    #range to box edge (according)
    #min_dis = np.where(mask.any(axis=-1), mask.argmax(axis=-1), 0) #left (low) border
    #max_dis = np.where(mask.any(axis=-1), mask[...,::-1].argmax(axis=-1), 0) #right (high) border

    #range to box edge (low)
    #min_dis = np.where(mask.any(axis=-1), mask.argmax(axis=-1), 0) #left (low) border
    #max_dis = np.where(mask.any(axis=-1), y_c*2-1-mask[...,::-1].argmax(axis=-1), 0) #right (high) border

    #silhouette only
    #min_dis = np.where(mask[...,:y_c].any(axis=-1), 1, 0)
    #max_dis = np.where(mask[...,y_c:].any(axis=-1), 1, 0)

    return np.stack([min_dis, max_dis], axis=-1)
    
    #depth image
    #min_dis = np.where(mask.any(axis=-1), mask.argmax(axis=-1), 0) #left (low) border
    #return min_dis #[T, z, x]


def farthest_point_sampling(points, k, device=torch.device('cpu')):
    '''
    points:  [(fix_dim), n, p] (x, y, z, ...)
    output: [k]
    '''
    coors = points[...,:3]
    if isinstance(coors, np.ndarray):
        dtype = 'ndarray'
        coors = torch.from_numpy(coors)
    else:
        dtype = 'tensor'
    if len(coors.shape) == 2:
        coors = coors.unsqueeze(0)
    dist_min = torch.full((points.shape[1],), float('inf'), device=device)
    id_now = np.random.randint(0,points.shape[1])
    ids = [id_now]
    for i in range(k-1):
        dist = torch.sum(coors**2, -1) + torch.sum(coors[:,id_now,:]**2) - 2*coors.matmul(coors[:,id_now,:])
        dist = torch.sqrt(F.relu(dist))  # [n]
        dist_min = torch.min(torch.stack([dist_min, dist]), dim=0)
        id_now = torch.argmax(dist_min)
        ids.append(id_now)
    return ids

def temporal_differ(seq, reduction=None):
    tmp_differ = [abs(seq[i+1]-seq[i]) for i in range(seq.shape[0]-1)]
    print('Processing length {} sequence'.format(len(seq)))
    if reduction is None:
        return np.stack(tmp_differ)
    else:
        if reduction == 'mean':
            return sum(tmp_differ)/len(tmp_differ)
        if reduction == 'sum':
            return sum(tmp_differ)
        else:
            raise NotImplementedError('Reduction {} is not supported'.format(reduction))

if __name__ == '__main__':
    inarr = torch.randn([8,128,256]) #[batch, n, p]
    sampled_ids = farthest_point_sampling(inarr, 16)
    print(sampled_ids.shape)
    ########################################################
