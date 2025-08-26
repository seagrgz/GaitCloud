import torch
import time
import warnings
import math
import torch.nn as nn
import torch.nn.functional as F
from models.my_resnet import BasicBlock, ResNet
#from pytorch_metric_learning import losses, miners
from collections import OrderedDict
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import numpy as np
from util.ssim import ssim

class ResBlock(nn.Module):
    def __init__(self, channels, win_size=3, stride=1, downsample=None, sample_dim=3):
        self.stride, self.channels = stride, channels
        super().__init__()
        if sample_dim == 2:
            self.Conv = nn.Conv2d
            self.norm_func = nn.BatchNorm2d
        elif sample_dim == 3:
            self.Conv = nn.Conv3d
            self.norm_func = nn.BatchNorm3d
        else:
            raise NotImplementedError(sample_dim)
        if isinstance(win_size, int):
            win_size = [win_size]*sample_dim
        self.bn1 = self.norm_func(channels[1])
        self.bn2 = self.norm_func(channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = self.Conv(channels[0], channels[1], win_size, stride=stride, padding=[int((k-1)/2) for k in win_size])
        self.conv2 = self.Conv(channels[1], channels[1], win_size, stride=1, padding='same')
        if downsample == None:
            self.downsample = nn.Sequential(
                    self.Conv(channels[0], channels[1], 1, stride=stride),
                    self.norm_func(channels[1]))
                    #nn.MaxPool3d(kernel_size=stride, stride=stride))
        else:
            self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride != 1 or self.channels[0] != self.channels[1]:
            identity = self.downsample(x)
            out += identity
        else:
            out += identity
        out = self.relu(out)
        return out

class ResMLP(nn.Module):
    def __init__(self, channels, win_size=1):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=win_size),
            nn.BatchNorm1d(channels[1]),
            self.act)
        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels=channels[1], out_channels=channels[1], kernel_size=win_size),
            nn.BatchNorm1d(channels[1]))

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)

class FlatIn(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channel, out_channel, 3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        '''
        x   : [n, c, h, w, l]
        out : [n, c, h, w, l]
        '''
        x = self.conv_in(x)
        x = self.bn_in(x)
        out = self.relu(x)
        return out

class LayerCNN(nn.Module):
    def __init__(self, in_channel=1, channels=[16,32], resblock=False):
        self.resblock = resblock
        super().__init__()

        self.conv_in = nn.Conv3d(in_channel, channels[0], (1,3,3), stride=1, padding=(0,1,1), bias=False)
        self.bn_in = nn.BatchNorm3d(channels[0])
        self.conv1 = nn.Conv3d(channels[0], channels[1], (1,3,3), stride=1, padding=(0,1,1))
        self.bn1 = nn.BatchNorm3d(channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels[1], channels[1], (1,3,3), stride=1, padding=(0,1,1))
        self.bn2 = nn.BatchNorm3d(channels[1])
        if resblock:
            if channels[0]!=channels[1]:
                self.shortcut = nn.Sequential(
                        nn.Conv3d(channels[0], channels[1], 1, stride=1),
                        nn.BatchNorm3d(channels[1]))
            else:
                self.shortcut = nn.Identity()

    def forward(self, x):
        '''
        x   : [n, c, h, w, l]
        out : [n, c, h, w, l]
        '''
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu(x)

        if self.resblock:
            identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.resblock:
            out += identity
        out = self.relu(out)
        return out

class LE_2D(nn.Module):
    def __init__(self, in_channel=1, channels=[16,32], resblock=False):
        self.resblock = resblock
        super().__init__()
        self.conv_in = nn.Conv2d(in_channel, channels[0], (1,3), stride=1, padding=(0,1), bias=False)
        self.bn_in = nn.BatchNorm2d(channels[0])
        self.conv1 = nn.Conv2d(channels[0], channels[1], (1,3), stride=1, padding=(0,1))
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels[1], channels[1], (1,3), stride=1, padding=(0,1))
        self.bn2 = nn.BatchNorm2d(channels[1])
        if resblock:
            if channels[0]!=channels[1]:
                self.shortcut = nn.Sequential(
                        nn.Conv2d(channels[0], channels[1], 1, stride=1),
                        nn.BatchNorm2d(channels[1]))
            else:
                self.shortcut = nn.Identity()

    def forward(self, x):
        '''
        x   : [n, c, h, w, l]
        out : [n, c, h, w, l]
        '''
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu(x)

        if self.resblock:
            identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.resblock:
            out += identity
        out = self.relu(out)
        return out

class Unet(nn.Module):
    def __init__(self):
        super().__init__()

    def create_down(self, in_channel, out_channel):
        block = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True)
            )
        return block

    def create_up(self, in_channel, out_channel):
        block = nn.Sequential(
            nn.Conv3d(in_channel*2, in_channel, 3, padding=1),
            nn.BatchNorm3d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channel, in_channel, 3, padding=1),
            nn.BatchNorm3d(in_channel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channel, out_channel, kernel_size=2, stride=2)
            )
        return block

    def create_trans(self, in_channel, trans_channel):
        block = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(in_channel, trans_channel, 3, padding=1),
            nn.BatchNorm3d(trans_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(trans_channel, trans_channel, 3, padding=1),
            nn.BatchNorm3d(trans_channel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(trans_channel, in_channel, kernel_size=2, stride=2)
            )
        return block

class GaitNorm(nn.Module):
    def __init__(self, io_channel, part_num=16):
        self.part_num = part_num
        super().__init__()
        self.unet = Unet()
        self.down1 = self.unet.create_down(64, 128)
        self.down2 = self.unet.create_down(128, 256)
        #self.down3 = unet.create_down(256,512)
        self.trans = self.unet.create_trans(256, 512)
        #self.up3 = unet.create_up(512, 256)
        self.up2 = self.unet.create_up(256, 128)
        self.up1 = self.unet.create_up(128, 64)
        self.outconv = nn.Sequential(
                nn.Conv3d(128, 64, 3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 64, 3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                )
        self.relu = nn.ReLU(inplace=True)
        #self.outconv = nn.Sequential(
        #        nn.Conv3d(64, 64, 1),
        #        nn.BatchNorm3d(64),
        #        nn.ReLU(inplace=True)
        #        )
        #self.cosim = nn.CosineSimilarity()

        #self.vis_feats = {}

    def forward(self, x, batch_size, training=True):
        '''
            x   : [2n/n, c, h, w, l]
            out : [n, c, h, w, l]
        '''
        n = x.shape[0]

        if training:
            ref = x[batch_size:]
            x = x[:batch_size]

        identity1 = x
        x = self.down1(x) #[n, 2c, h/2, w/2, l/2]
        identity2 = x
        x = self.down2(x) #[n, 4c, h/4, w/4, l/4]
        identity3 = x
        out = self.trans(x) #[n, 4c, h/4, w/4, l/4]
        out = self.up2(torch.cat([identity3, out], dim=1)) #[n, 2c, h/2, w/2, l/2]
        out = self.up1(torch.cat([identity2, out], dim=1)) #[n, c, h, w, l]
        out = self.outconv(torch.cat([identity1, out], dim=1)) #[n, c, h, w, l]
        #out = self.up2(out)
        #out = self.up1(out)

        #identity1 = out
        #out = self.outconv(out)
        #out = self.relu(out+identity1)

        #feat = feats[:batch_size]
        #ref = feats[batch_size:]
        if training:
            assert n == batch_size*2, "Number of samples ({}) and references ({}) should be identical".format(batch_size, n-batch_size)
            penalty = self.get_penalty(out, ref)
        else:
            penalty = 0
        return out, penalty

    def get_penalty(self, feat, ref):
        #n, c = feat.shape[:2]
        #feat = torch.mean(feat.view(n, c, part_num, -1), dim=-1) #[n+ref, 4c, p]

        #penalty = self.cosim(feat[:batch_size], feat[batch_size:]).transpose(0,1) #[p, n]
        #dists = self.ComputeDistance(feat, batch_size) #[p, n*ref]
        #penalty = self.Reducer(dists) #[1]
        feat, ref = self.Partition(feat), self.Partition(ref) #default setup: [4,2,2]
        penalty = self.SSIM(feat, ref, partition=True)
        return penalty

    def Reducer(self, penalty):
        eps = 10e-9
        p_sum = penalty.sum(-1) #[p]
        num = (penalty.detach().clone() != 0).sum(-1).float()
        p_mean = p_sum/(num+eps)
        p_mean[num == 0] = 0
        return p_mean.mean()

    def NonzeroMAE(self, feat, ref, eps=1e-6):
        differ = torch.abs(feat-ref)
        nonzero_d = differ[differ>eps]
        if len(nonzero_d) > 0:
            return (torch.mean(nonzero_d) + torch.max(differ.view(feat.shape[0], -1), -1)[0].mean())/2
        else:
            return torch.tensor(0).type_as(feat)

    def ComputeDistance(self, feat, batch_size):
        """
            feat: [n+ref, c, p]
            dist: [p, n*ref]
        """
        eps = 1e-9
        feat = feat.permute(2, 0, 1) #[p, n+ref, c]
        x = feat[:, :batch_size, :]
        y = feat[:, batch_size:, :]
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, ref]
        inner = x.matmul(y.transpose(1, 2))  # [p, n, ref]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist) + eps)  # [p, n, ref]
        return dist.view(dist.shape[0], -1)

    def SSIM(self, x, y, win_size=5, win_sigma=1.5, partition=False):
        '''
        x/y     : [n, c, h, w, l]
        output  : []
        '''
        L = max(x.max(),y.max())-min(x.min(),y.min())
        with torch.no_grad():
            output = 1 - ssim(x, y, data_range=L, size_average=False, win_size=win_size, win_sigma=win_sigma)
        if output.min() < 0:
            warnings.warn('Got a negtive value ({}) in SSIM loss!'.format(output.min()))
        if partition:
            #return output.max()
            return output.sum()
        else:
            return output.mean()

    def Partition(self, embeds, parts=[4,2,2]):
        '''
        embeds: [n, c, h, w, l]
        output: [p, n*c, h, w, l]
        '''
        shape = embeds.shape
        assert len(shape) == 5, 'Input should have shape of [n, c, h, w, l], but got {} dimensions'.format(len(embeds.shape))
        n, c, h, w, l = shape
        seg_z, seg_x, seg_y = [int(shape[i-3]/parts[i]) for i in range(3)]
        assert (h == seg_z*parts[0]) and (w == seg_x*parts[1]) and (l == seg_y*parts[2]), 'Embeding size ({}) should be divisable by partition number ({}) in all dimensions'.format([h, w, l], parts)
        part_embeds = torch.stack([embeds[...,seg_z*z:seg_z*(z+1),seg_x*x:seg_x*(x+1),seg_y*y:seg_y*(y+1)] for y in range(parts[2]) for x  in range(parts[1]) for z in range(parts[0])])
        return part_embeds.view(-1,n*c,seg_z,seg_x,seg_y) #[p, n*c, h/p_z, w/p_x, l/p_y]

class ScaleFusionCNN(nn.Module):
    def __init__(self, in_channel=32, out_channel=64, batch_size=8):
        self.centroid = 18
        super().__init__()
        #self.scale = nn.Parameter(torch.tensor([0.5], device='cuda')) #learnable
        self.norm_func = nn.BatchNorm3d
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(2, stride=2)
        #self.dropout = nn.Dropout(0.5, inplace=True)
        self.basicbranch = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, stride=1, padding=1),
                self.norm_func(out_channel),
                self.relu,
                nn.Conv3d(out_channel, out_channel, 3, stride=1, padding=1),
                self.norm_func(out_channel))
                #self.pool)
        self.scaledbranch = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 7, stride=1, padding=3),
                self.norm_func(out_channel),
                self.relu,
                nn.Conv3d(out_channel, out_channel, 7, stride=1, padding=3),
                self.norm_func(out_channel))
                #self.pool)
        if in_channel!=out_channel:
            self.shortcut = nn.Sequential(
                    nn.Conv3d(in_channel, out_channel, 1),
                    self.norm_func(out_channel))
                    #self.pool)
        else:
            self.shortcut = nn.Identity()#self.pool
        
    def get_scale(self, positions):
        return (positions/(positions+self.centroid))

    def forward(self, x, positions):
        '''
        x   : [n, c_in, h, w, l]
        out : [n, c_out, h, w, l]
        '''
        batch_size = x.shape[0]

        identity = self.shortcut(x)
        scaled_x = self.scaledbranch(x)
        x = self.basicbranch(x)

        scale = self.get_scale(positions).view(batch_size,1,1,1,1) #position dynamic
        #scale = torch.full((batch_size,1,1,1,1), 0.5, device='cuda') #fixed
        #scale = self.scale.repeat(batch_size,1,1,1,1) #learnable
        out = scaled_x * scale + x * (1 - scale) + identity
        out = self.relu(out)

        return out

class ResNet9_3D(ResNet):
    def __init__(self, in_channels = 1, inplanes=None, maxpool=False):
        self.maxpool_flag = maxpool
        block = BasicBlock
        channels = [64,128,256,512]
        layers = [1,1,1,1]
        strides = [1,2,2,1]

        super(ResNet9_3D, self).__init__(block, layers)

        if inplanes is None:
            self.inplanes = channels[0]
        else:
            self.inplanes = inplanes
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv3d(in_channels, self.inplanes, 3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=strides[0], dilate=False)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=strides[1], dilate=False)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=strides[2], dilate=False)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=strides[3], dilate=False)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if blocks >= 1:
            layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        else:
            def layer(x): return x
        return layer

    def forward(self, x):
        """
        x   : [n, h, w, l] / [n, c, h, w, l]
        out : [n, c_out, h_out, w_out, l_out]
        """
        if len(x.shape) == 4:
            x = self.conv1(x.unsqueeze(1))
            x = self.bn1(x)
            x = self.relu(x)
            if self.maxpool_flag:
               x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.layer4(x)
        return out

class HPP():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=[16]):
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, ...]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.reshape(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)

class SPP():
    """
        Side Pyramid Pooling
    """

    def __init__(self, bin_num=[10]):
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w, l]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        x = torch.permute(x, (0,1,3,4,2)).contiguous() #[n, c, w, l, h]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)

#class ConvHPP(nn.Module):
#    def __init__(self, bin_num, channel, win_size):
#        self.bin_num = bin_num
#        super(ConvHPP, self).__init__()
#        self.conv_mask = nn.ModuleDict({})
#        for b in bin_num:
#            self.conv_mask['bin{}'.format(b)] = nn.Sequential(
#                    nn.Conv3d(channel, channel, kernel_size=win_size, stride=(win_size[0], 1, 1), bias=False),
#                    #nn.Conv3d(1, 1, kernel_size=win_size, stride=(win_size[0], 1, 1), bias=False),
#                    nn.BatchNorm3d(channel),
#                    nn.ReLU())
#
#    def __call__(self, x):
#        """
#            x  : [n, c, h, w, l]
#            ret: [n, c, p] 
#        """
#        n, c = x.size()[:2]
#        features = []
#        for b in self.bin_num:
#            #z = self.conv_mask['bin{}'.format(b)](x.view(n*c,1,*x.size()[2:]))
#            z = self.conv_mask['bin{}'.format(b)](x)
#            z = z.squeeze().view(n,c,b) + torch.max(x.view(n,c,b,-1), -1)[0]
#            features.append(z)
#        return torch.cat(features, -1)

class ConvHPP(nn.Module):
    def __init__(self, bin_num, channel, win_size):
        self.bin_num = bin_num
        super(ConvHPP, self).__init__()
        self.conv_mask = nn.ModuleDict({})
        if len(win_size) == 3:
            conv = nn.Conv3d
            norm = nn.BatchNorm3d
            stride = (win_size[0], 1, 1)
        elif len(win_size) == 2:
            conv = nn.Conv2d
            norm = nn.BatchNorm2d
            stride = (win_size[0], 1)
        else:
            raise NotImplementedError('Window size should be a 2/3-dimension tuple or list!')

        for b in bin_num:
            self.conv_mask['bin{}'.format(b)] = nn.Sequential(
                    conv(channel, channel, kernel_size=win_size, stride=stride, bias=False),
                    norm(channel),
                    nn.ReLU())

    def __call__(self, x):
        """
            x  : [n, c, h, w, l]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = self.conv_mask['bin{}'.format(b)](x)
            z = z.squeeze().view(n,c,b) + torch.max(x.view(n,c,b,-1), -1)[0]
            features.append(z)
        return torch.cat(features, -1)

class LinearProjection(nn.Module):
    def __init__(self, parts_in=10, parts_out=16, channels=512):
        super(LinearProjection, self).__init__()
        self.fc_bin = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(channels, parts_in, parts_out)))

    def forward(self, x):
        """
            x: [n, c, p_in]
            out: [n, c, p_out]
        """
        x = x.permute(1, 0, 2).contiguous() #[c, n, p_in]
        out = x.matmul(self.fc_bin) #[c, n, p_out]
        return out.permute(1, 0, 2).contiguous()

class SeparateBNNecks(nn.Module):
    """
        Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num=16, in_channels=256, class_num=250, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)

class SeparateFCs(nn.Module):
    def __init__(self, parts_num=16, in_channels=512, out_channels=256, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [n, c_in, p]
            out: [n, c_out, p]
        """
        x = x.permute(2, 0, 1).contiguous()
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out.permute(1, 2, 0).contiguous()

class SeparateBNNecks(nn.Module):
    """
        Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num=16, in_channels=256, class_num=250, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [n, c, p]
        """
        if self.parallel_BN1d:
            n, c, p = x.size()
            x = x.view(n, -1)  # [n, c*p]
            x = self.bn1d(x)
            x = x.view(n, c, p)
        else:
            x = torch.cat([bn(_x) for _x, bn in zip(x.split(1, 2), self.bn1d)], 2)  # [p, n, c]
        feature = x.permute(2, 0, 1).contiguous()
        if self.norm:
            feature = F.normalize(feature, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            logits = feature.matmul(self.fc_bin)
        return feature.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()

class SymbolMHA(nn.Module):
    def __init__(self, embed_dim=512, num_head=1):
        super().__init__()
        self.MHA = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_head, batch_first=True)

    def forward(self, x, symbol):
        """
            x:      [n, c, h, w, l]
            symbol: ibid
            out:    [n, c, h, w, l]
        """
        assert (symbol.shape == x.shape), 'input and symbol should have same shape'
        n, c, h, w, l = x.shape
        x = x.view(n, c, -1).transpose(1, 2) #[n, h*w*l, c]
        symbol = symbol.view(n, c, -1).transpose(1, 2)
        attention_out = self.MHA(x, symbol, x, need_weights=False)[0] #[n, h*w*l, c]
        return attention_out.transpose(1, 2).contiguous().view(n, c, h, w, l)

class SymbolAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, symbol):
        assert (symbol.shape == x.shape), 'input and symbol should have same shape'
        n, c, h, w, l = x.shape
        x = x.view(n, c, h, -1).permute(0, 2, 3, 1) #[n, h, w*l, c]
        symbol = symbol.view(n, c, h, -1).permute(0, 2, 3, 1)
        attention_out = torch.cat(
                [F.scaled_dot_product_attention(x[:,i,...], symbol[:,i,...], x[:,i,...]).unsqueeze(1) for i in range(h)],
                dim=1)
        return attention_out.permute(0, 3, 1, 2).contiguous().view(n, c, h, w, l)

class TemporalAttention(nn.Module):
    def __init__(self, in_channel=None, out_channel=None, dk=None):
        super().__init__()
        self.in_ch, self.out_ch, self.dk = in_channel, out_channel, dk
        if not ((self.in_ch is None) and (self.out_ch is None) and (self.dk is None)):
            self.trans = True
            assert self.in_ch is not None
            assert self.out_ch is not None
            assert self.dk is not None
            self.qkin_conv = nn.Conv1d(self.in_ch, self.dk, 1)
            self.out_conv = nn.Conv1d(self.dk, self.out_ch, 1)
        else:
            self.trans = False

    def forward(self, q, kv):
        '''
        q:  [D_fix, c, L_q]
        kv:  [D_fix, c, L_kv]
        out:[D_fix, c, L_k]
        '''
        L_q, L_kv = q.shape[2], kv.shape[2]
        if self.trans:
            kv, q = torch.split(self.qkin_conv(torch.cat([kv,q], dim=2)), [L_kv,L_q], dim=2)
            k, v = kv, kv
        else:
            q, k, v = q, kv, kv
            self.dk = q.shape[1]

        logits = torch.matmul(q.transpose(1,2), k) #[D_fix, L_q, L_kv]
        weights = F.softmax(logits/math.sqrt(self.dk), dim=-1)
        out = torch.matmul(weights, v.transpose(1,2)) #[D_fix, L_q, dk]

        if self.trans:
            out = self.out_conv(out.transpose(1,2)) #[D_fix, out_ch, L_k]
        else:
            out = out.transpose(1,2)

        return out, weights

def PCA_image(embed, n_components=3):
    """
        embed:  [c, ...]
        im_out: [..., 3]
    """
    if not torch.is_tensor(embed):
        embed = torch.from_numpy(embed)
    shape = embed.shape[1:]
    c = embed.shape[0]
    embed = embed.reshape(c, -1).transpose(0, 1) #[feat, c]
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(embed) #[feat, 3]
    im_out = minmax_scale(features, (0,225), axis=1)
    return np.reshape(im_out, list(shape)+[3])

class LossAggregator(nn.Module):
    def __init__(self, margin, scale, lamda) -> None:
        super().__init__()
        #self.TPloss = losses.TripletMarginLoss()
        #self.CEloss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.TPloss = TripletLoss(margin=margin, scale=scale, lamda=lamda, loss_term_weight=1)
        self.CEloss = CrossEntropyLoss(scale=16, loss_term_weight=0.1)

        self.TP_weight = 1#nn.Parameter(torch.Tensor(1))
        self.CE_weight = 0.1#nn.Parameter(torch.Tensor(1))
        self.Norm_weight = 1
        #self.miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="all")
        self.scale = 1

    def forward(self, TP_feat, CE_feat, labels, Norm_loss=torch.tensor(0)):
        #TP_feat = TP_feat.view(TP_feat.shape[0], -1)
        #d = CE_feat.shape[-1]

        #indices_tuple = self.miner(TP_feat, labels)
        #TP_loss = self.TPloss(TP_feat, labels)
        #CE_loss = self.CEloss(CE_feat*self.scale, labels.unsqueeze(1).repeat(1, d).long())
        #mined_triplets = self.miner.num_triplets

        #st = time.time()
        loss_sum = .0
        TP_loss, loss_num = self.TPloss(TP_feat, labels)
        #tp_end = time.time()
        CE_loss = self.CEloss(CE_feat, labels)
        #ce_end = time.time()
        mined_triplets = loss_num.sum().item()

        #print(penalty)
        loss_sum = TP_loss.mean()*self.TP_weight + CE_loss.mean()*self.CE_weight + Norm_loss*self.Norm_weight
        #loss_sum = TP_loss.mean()*(1+Norm_loss)*self.TP_weight + CE_loss.mean()*self.CE_weight
        #end = time.time()
        #print('tp time: {}, ce time: {}, total: {}'.format(round(tp_end-st, 4), round(ce_end-tp_end, 4), round(end-st, 4)))
        return [loss_sum, TP_loss.mean(), CE_loss.mean(), Norm_loss], mined_triplets

class Odict(OrderedDict):
    def append(self, odict):
        dst_keys = self.keys()
        for k, v in odict.items():
            if not is_list(v):
                v = [v]
            if k in dst_keys:
                if is_list(self[k]):
                    self[k] += v
                else:
                    self[k] = [self[k]] + v
            else:
                self[k] = v

class BaseLoss(nn.Module):
    """
    Base class for all losses.

    Your loss should also subclass this class.
    """

    def __init__(self, loss_term_weight=1.0):
        """
        Initialize the base class.

        Args:
            loss_term_weight: the weight of the loss term.
        """
        super(BaseLoss, self).__init__()
        self.loss_term_weight = loss_term_weight

    def forward(self, logits, labels):
        """
        The default forward function.

        This function should be overridden by the subclass. 

        Args:
            logits: the logits of the model.
            labels: the labels of the data.

        Returns:
            tuple of loss and info.
        """
        return .0

class CrossEntropyLoss(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1, loss_term_weight=1.0):
        super(CrossEntropyLoss, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps

    def forward(self, logits, labels):
        """
            logits: [n, c, p]
            labels: [n]
        """
        n, c, p = logits.size()
        logits = logits.float()
        labels = labels.unsqueeze(1)
        if self.label_smooth:
            loss = F.cross_entropy(
                logits*self.scale, labels.repeat(1, p).long(), label_smoothing=self.eps)
        else:
            loss = F.cross_entropy(logits*self.scale, labels.repeat(1, p))
        return loss

class TripletLoss(BaseLoss):
    def __init__(self, margin, scale, lamda, loss_term_weight=1.0):
        super(TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin
        self.scale = scale
        self.lamda = lamda
        self.balance = np.power(margin, 1/lamda)/margin

    #@gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [n, c, p], label: [n]
        #st = time.time()
        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]
        
        ref_embed, ref_label = embeddings, labels
        #dist_st = time.time()
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        #dist_ed = time.time()
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
        #triplet_ed = time.time()
        dist_diff = (ap_dist - an_dist).view(dist.size(0), -1) #[p, ...]
        #differ_ed = time.time()
        loss = torch.pow(self.balance * F.relu(dist_diff + self.margin), self.lamda) * self.scale

        loss_avg, loss_num = self.AvgNonZeroReducer(loss)
        #ed = time.time()
        #print('dist time: {}, triplet time: {}, loss_time: {}, total: {}'.format(round(dist_ed-dist_st, 4), round(triplet_ed-dist_ed, 4), round(ed-triplet_ed, 4), round(ed-st, 4)))
        #print('triplet time: {}, differ time: {}'.format(round(triplet_ed-dist_ed, 4), round(differ_ed-triplet_ed, 4)))

        return loss_avg, loss_num.detach()

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss.detach().clone() != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        eps = 1e-9
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist) + eps)  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        #st = time.time()
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        matches = matches.fill_diagonal_(False)
        #index_ed = time.time()
        p, n, _ = dist.size()
        #matches_id = matches.nonzero(as_tuple=False)
        #diffenc_id = diffenc.nonzero(as_tuple=False)
        #ap_dist = dist[:, matches_id[:,0], matches_id[:,1]].view(p,n,-1,1)
        #an_dist = dist[:, diffenc_id[:,0], diffenc_id[:,1]].view(p,n,1,-1)
        ap_dist = dist[:, matches].view(p, -1, 1)
        an_dist = dist[:, diffenc].view(p, 1, -1)
        #ed = time.time()
        #print('map time: {}, extract time: {}'.format(round(index_ed-st, 4), round(ed-index_ed, 4)))
        return ap_dist, an_dist
