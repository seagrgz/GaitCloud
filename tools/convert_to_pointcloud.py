#!/usr/bin/env python
#point data is formed as [x, y, z, intensity, distance, azimuth, ID]

import rclpy
import argparse
import struct
import time
import math
import signal
import numpy as np
from rclpy.node import Node
from velodyne_msgs.msg import VelodyneScan
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from config.load_config import load_yaml

#referance of vertical angle (degree)
va = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
#vertical correction (mm)
vc = [11.2, -0.7, 9.7, -2.2, 8.1, -3.7, 6.6, -5.1, 5.1, -6.6, 3.7, -8.1, 2.2, -9.7, 0.7, -11.2]
parser = argparse.ArgumentParser(description='Raw LiDAR data converter to numpy PCD')
parser.add_argument('--cfgs', type=str, default='config/config.yaml', help='Path to yaml config file')
opt = parser.parse_args()

class Recorder(Node):
    def __init__(self, lifetime, save_path, save_pt):
        super().__init__('recorder')
        self.raw_subscriber = self.create_subscription(VelodyneScan, 'velodyne_packets', self.callback, 30)
        self.pub = self.create_publisher(PointCloud2, 'frame_xyz', 10)
        self.field = [PointField(name = 'x', offset = 0, datatype = 7, count = 1),
                        PointField(name = 'y', offset = 4, datatype = 7, count = 1),
                        PointField(name = 'z', offset = 8, datatype = 7, count = 1),
                        PointField(name = 'intensity', offset = 12, datatype = 2, count = 1),
                        PointField(name = 'distance', offset = 13, datatype = 7, count = 1),
                        PointField(name = 'azimuth', offset = 17, datatype = 7, count = 1),
                        PointField(name = 'ID', offset = 21, datatype = 2, count = 1)]
        self.header = Header(frame_id = 'velodyne')
        
        self.frame = []
        self.sample = []
        self.frame_end = 0
        self.frame_count = 0
        self.ex_azi = -1

        #recorder parameters
        self.lifetime, self.pth, self.save_pt = lifetime, save_path, save_pt

    def callback(self, package):
        packets = package.packets
        self.get_logger().info('Receving packets ...')
        for packet in packets:
            data = packet.data
            timestamp = packet.stamp.sec + packet.stamp.nanosec/(10^9)
            self.main_process(data)
            
    def main_process(self, data):
        '''
        conver velodyne raw data to point cloud
        '''
        pointcloud = []
        for block in range(12):
            pos_azi = block*100+2 #index of azimuth data
            azimuth = struct.unpack('H', struct.pack('BB', data[pos_azi], data[pos_azi+1]))[0]
            #start a new round if azimuth smaller then last block
            if azimuth < self.ex_azi:
                self.frame_end = 1
            self.ex_azi = azimuth
            #get azimuth gap between two firing sequences to calculate azimuth for each point
            if block < 11:
                azimuth_n=struct.unpack('H', struct.pack('BB', data[pos_azi+100], data[pos_azi+101]))[0]
                if azimuth_n < azimuth:
                    azigap = azimuth_n+36000-azimuth
                else:
                    azigap = azimuth_n-azimuth

            for ch in range(32):
                pos_dis = block*100+ch*3+4 #index of distance data
                pos_ins = block*100+ch*3+6 #index of intensity data
                distance = struct.unpack('H', struct.pack('BB', data[pos_dis], data[pos_dis+1]))[0]*2
                Id = ch%16
                intensity = data[pos_ins]
                #get azimuth for each point
                if ch < 16:
                    azi = int(azimuth+(azigap*Id/48))
                else:
                    azi = int(azimuth+azigap*(Id/48+0.5))
                point_data = self.to_xyz(Id, distance, intensity, azi)
                if point_data != None:
                    self.frame.append(point_data)

            if self.frame_end == 1: #cut the frame when starting a new round
                #print(np.count_nonzero(self.frame, axis=0))
                if len(self.frame) > 0:
                    self.sample.append(np.array(self.frame))
                    self.get_logger().info('Frame {} converted'.format(self.frame_count))
                self.frame_end = 0
                if self.frame_count == self.lifetime and self.save_pt:
                    self.save_to_npy()
                    print('[INFO] Totally {} frames saved to {}'.format(self.frame_count, self.pth))
                    self.frame_count = 0
                    exit(0);
                elif not self.save_pt:
                    self.publish(self.frame)
                    self.sample = []
                self.frame = []
                self.frame_count += 1

    def to_xyz(self, ID, dis, ins, azi):
        '''
        convert single point to Cartesian coordinate
        '''
        global va, vc
        if dis > 0:
            R = dis/1000
            omega = math.radians(va[ID])
            alpha = math.radians(azi/100)
            x = float(R*math.cos(omega)*math.sin(alpha))
            y = float(R*math.cos(omega)*math.cos(alpha))
            z = float(R*math.sin(omega)+vc[ID]/1000)
            coor = [x, y, z, ins, R, azi/100, ID]
        else:
            coor = None
        return coor

    def publish(self, points):
        pubdata = point_cloud2.create_cloud(self.header, self.field, points)
        self.pub.publish(pubdata)
        print('[{}] Publishing frame {}'.format(time.time(), self.frame_count))

    def save_to_npy(self):
        print('[INFO] saving data ...')
        np.save(self.pth, np.asarray(self.sample, dtype=object), allow_pickle=True)
        self.get_logger().info('Saving point cloud to {}'.format(self.pth))

def quit_gracefully(signum, stack):
    print('[INFO] Node shutting down...')
    exit(0);

if __name__ == '__main__':
    signal.signal(signal.SIGINT, quit_gracefully)
    rclpy.init(args=None)
    cfgs = load_yaml(opt.cfgs)['pcd_converter']
    recorder = Recorder(lifetime=cfgs['lifetime'], save_path=cfgs['save_path'], save_pt=cfgs['save_pt'])
    print('[INFO] Node initialization complete')
    try:
        print('Waiting for data\n...')
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        if recorder.frame_count != 0 and cfgs['save_pt']:
            recorder.save_to_npy()
        quit_gracefully()
