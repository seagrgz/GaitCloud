import torch
import torch.nn as nn
from models.module import ResNet9_3D, FlatIn, ConvBlock, HPP, ConvHPP, SeparateFCs, SeparateBNNecks, LossAggregator
import numpy as np

class Baseline(nn.Module):
    def __init__(self, args, in_size):
        super().__init__()
        #self.encoder = ResNet9_3D()
        self.encoder = nn.Sequential(
                FlatIn(64),
                ConvBlock([64,64], stride=1),
                ConvBlock([64,128]),
                ConvBlock([128,256]),
                ConvBlock([256,512], stride=1))
        self.FCs = SeparateFCs()
        self.BNNecks = SeparateBNNecks(class_num=len(args.target))
        self.HPP = HPP()
        self.mergeloss = LossAggregator(margin=0.2, scale=1, lamda=1)

    def forward(self, x, label=None, training=True, **kwargs):
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
            losses, mined_triplets = self.mergeloss(embed_tp, embed_ce, label)
            output = tuple([losses, mined_triplets, embed_tp])
        else:
            output = embed_tp

        return output
