import torch
import random
import os
import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset

from util.data_util import sa_create

class GaitDataset(Dataset):
    def __init__(self, split, data_root, args, datalist):
        super().__init__()
        self.args = args
        self.data_list, self.split, = datalist, split
        self.target = args.target

        #load data
        if args.structure == 'no':
            self.share_memory = False
            self.data_root = data_root
        else:
            self.share_memory = True
            for item in self.data_list:
                data_path = os.path.join(data_root, item + '.npy')
                data = np.load(data_path, allow_pickle=True)
                if not os.path.exists("/dev/shm/{}{}{}".format(self.args.identifier, self.args.data_split, item)):
                    sa_create("shm://{}{}{}".format(self.args.identifier, self.args.data_split, item), data)
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        name = '{}{}{}'.format(self.args.identifier, self.args.data_split, self.data_list[data_idx])
        if self.share_memory:
            packed_data = SA.attach("shm://{}".format(name), ro=True).copy()
        else:
            packed_data = np.load(os.path.join(self.data_root, self.data_list[data_idx]+'.npy'), allow_pickle=True)

        #use voxelized sample for input
        if self.args.data_split == 'SUS':
            if self.args.structure == 'LidarGait':
                data = packed_data
                position = 0
            else:
                data = packed_data[0]
                position = packed_data[1]
            #data = np.random.randint(0, 10, (50,50,20))
            #data[data!=0] = np.random.randint(0,10)
            #data[data!=0] = 1
            label = int(self.data_list[data_idx][-4:])
            #print(label)

        else:
            raise NotImplementedError('Data set {} is not implemented'.format(self.args.data_split))
        
        #print(label)
        #label mapping
        if (self.args.use_Aloss and self.split == 'train'):
            label = self.target.index(label)

        return data, label, [position, name]
