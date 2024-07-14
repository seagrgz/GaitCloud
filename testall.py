import torch.nn as nn
import time
from models.module import ConvBlock, FlatIn, LayerCNN, ScaleFusionCNN, HPP, ConvHPP, SeparateFCs, SeparateBNNecks, LossAggregator

class TestAll(nn.Module):
    def __init__(self, args, in_size):
        super().__init__()
        #self.layerencoder = LayerCNN(channels=[16,64], resblock=True)
        self.layerencoder = FlatIn(64)
        #self.scalencoder = ScaleFusionCNN(in_channel=64, out_channel=64)
        self.encoder = nn.Sequential(
                ConvBlock([64,64], stride=1),
                ConvBlock([64,128]),
                ConvBlock([128,256]),
                ConvBlock([256,512], stride=1))
        self.FCs = SeparateFCs()
        self.BNNecks = SeparateBNNecks(class_num=len(args.target))
        self.HPP = HPP()
        #self.HPP = ConvHPP()
        self.mergeloss = LossAggregator(margin=0.2, scale=1, lamda=1)

    def forward(self, x, labels=None, training=True, positions=None):
        #st = time.time()
        x = self.layerencoder(x)
        #layer_t = time.time()
        #x = self.scalencoder(x, positions.round())
        #scale_t = time.time()
        out = self.encoder(x)
        #encode_t = time.time()

        #Horizontal Pyramid Pooling
        feat = self.HPP(out)
        #hpp_t = time.time()

        #dense
        embed_tp = self.FCs(feat)
        #embed_t = time.time()

        if training:
            #BNNeck for classification
            _, embed_ce = self.BNNecks(embed_tp)
            
            #loss aggregation
            losses, mined_triplets = self.mergeloss(embed_tp, embed_ce, labels)
            #loss_t = time.time()
            output = tuple([losses, mined_triplets, embed_tp])
        else:
            output = embed_tp
            #loss_t = time.time()
        out_t = time.time()
        #print('loss:{}, out:{}'.format(round(loss_t-embed_t, 4), round(out_t-loss_t, 4)))

        return output
