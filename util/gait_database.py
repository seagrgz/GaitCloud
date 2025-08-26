import torch
import random
import os
import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset

from util.data_util import sa_create

class GaitDataset(Dataset):
    def __init__(self, split, data_root, args, datalist, share_memory=True):
        super().__init__()
        self.args = args
        self.data_list, self.split, = datalist, split
        self.target = args.target
        self.share_memory = share_memory

        #load data
        if not share_memory:
            self.data_root = data_root
        else:
            for item in self.data_list:
                data_path = os.path.join(data_root, item)
                data = np.load(data_path)
                if isinstance(data, np.lib.npyio.NpzFile):
                    data = np.asarray([data['main'], data['addons']], dtype=object)
                if not os.path.exists("/dev/shm/{}{}{}".format(self.args.identifier, self.args.data_name, item)):
                    sa_create("shm://{}{}{}".format(self.args.identifier, self.args.data_name, item), data)
        print("Totally {} samples in {} set.".format(len(self.data_list), split))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        name = '{}{}{}'.format(self.args.identifier, self.args.data_name, self.data_list[idx])
        if self.share_memory:
            packed_data = SA.attach("shm://{}".format(name), ro=True).copy()
        else:
            data_path = os.path.join(self.data_root, self.data_list[idx])
            packed_data = np.load(data_path)
            if isinstance(packed_data, np.lib.npyio.NpzFile):
                packed_data = np.asarray([packed_data['main'], packed_data['addons']], dtype=object)

        #use voxelized sample for input
        if packed_data.dtype == object:
            data = packed_data[0]
            addons = packed_data[1]
        else:
            data = packed_data
            addons = 0
        label = int(self.data_list[idx][:4]) #label first

        #print(label)
        #label smooth
        if (self.args.use_Aloss and self.split in ['train', 'ref']):
            label = self.target.index(label)

        return data, label, [addons, self.data_list[idx][:-4]]
