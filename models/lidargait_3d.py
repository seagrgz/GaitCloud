#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.my_resnet import BasicBlock, ResNet
from models.module import ResNet9_3D, HPP, SeparateFCs, SeparateBNNecks, LossAggregator
import numpy as np

class LidarGait3D(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feat_buff = []
        self.sample_L = args.frame_size

        self.encoder = ResNet9_3D()
        self.TP = PackSequenceWrapper(torch.max, args.frame_size)
        self.HPP = HPP()
        self.FCs = SeparateFCs()
        self.BNNecks = SeparateBNNecks(class_num=len(args.target))
        self.mergeloss = LossAggregator(margin=0.2, scale=1, lamda=1)

    def forward(self, x, label=None, training=False, **kwargs):
        '''
        input   : [n, t, h, w, l]
        output  : embedding[n, c, p]
        '''
        in_shape = x.shape
        n, t = in_shape[:2]
        #ResNet9
        feat = self.encoder(x.view(-1, *in_shape[2:])) #(n, t, h, w, l) -> (n*t, c, h, w, l)

        #Temporal pooling
        feat = self.TP(feat.view(n, t, *feat.shape[1:]).transpose(1,2), options={"dim": 2})[0] #(n, c, t, h, w, l)

        #Horizontal Pyramid Pooling
        feat = self.HPP(feat) #(n, c, h, w, l) -> (n, c, p)
        
        #dense
        embed_tp = self.FCs(feat)

        if training:
            #BNNeck for classification
            _, embed_ce = self.BNNecks(embed_tp)
            
            #loss aggregation
            losses, mined_triplets = self.mergeloss(embed_tp, embed_ce, label)
            output = tuple([losses, mined_triplets, embed_tp])
        else:
            output = embed_tp

        return output, output

class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func, seqL):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func
        self.seqL = [seqL]

    def forward(self, seqs, dim=2, options={}):
        """
            In  seqs: [n, c, s, ...]
            Out rets: [n, ...]
        """
        start = [0]
        seqL = self.seqL
        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(dim, curr_start, curr_seqL)
            rets.append(self.pooling_func(narrowed_seq, **options))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)

def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))
