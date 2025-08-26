import torch
import torch.nn as nn
from models.module import FlatIn, ResBlock, HPP, ConvHPP, SeparateFCs, SeparateBNNecks, LossAggregator
import numpy as np

class Baseline(nn.Module):
    def __init__(self, args, in_size):
        super().__init__()
        self.encoder = nn.Sequential(
                FlatIn(args.feature, 64),
                ResBlock([64,64], stride=1),
                ResBlock([64,128]),
                ResBlock([128,256]),
                ResBlock([256,512], stride=1))
        self.FCs = SeparateFCs()
        self.BNNecks = SeparateBNNecks(class_num=len(args.target))
        self.HPP = HPP()
        self.mergeloss = LossAggregator(margin=0.2, scale=1, lamda=1)

    def forward(self, x, labels=None, training=True, **kwargs):
        #ResNet9
        out = self.encoder(x)
        #print('encoder out: ', out.shape)

        #Horizontal Pyramid Pooling
        feat = self.HPP(out)
        #print('HPP out: ', feat.shape)

        #dense
        embed_tp = self.FCs(feat)
        #print('embeddings: ', embed_tp.shape)

        if training:
            #BNNeck for classification
            _, embed_ce = self.BNNecks(embed_tp)
            
            #loss aggregation
            losses, mined_triplets = self.mergeloss(embed_tp, embed_ce, labels)
            output = tuple([losses, mined_triplets, embed_tp])
        else:
            output = embed_tp

        return output, None
