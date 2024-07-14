import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from models.my_resnet import BasicBlock, ResNet
from pytorch_metric_learning import losses, miners
from collections import OrderedDict
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, channels, stride=2, downsample=None):
        self.stride, self.channels = stride, channels
        super().__init__()
        self.norm_func = nn.BatchNorm3d
        self.bn1 = self.norm_func(channels[1])
        self.bn2 = self.norm_func(channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(channels[0], channels[1], 3, stride=stride, padding=1)
        self.conv2 = nn.Conv3d(channels[1], channels[1], 3, stride=1, padding=1)
        if downsample == None:
            self.downsample = nn.Sequential(
                    nn.Conv3d(channels[0], channels[1], 1, stride=stride),
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

class FlatIn(nn.Module):
    def __init__(self, channels):
        in_channels = 1
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        '''
        x   : [n, h, w, l]
        out : [n, c, h, w, l]
        '''
        x = self.conv_in(x.unsqueeze(1))
        x = self.bn_in(x)
        out = self.relu(x)
        return out

class LayerCNN(nn.Module):
    def __init__(self, channels=[16,32], resblock=False):
        in_channel = 1
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
                        nn.Conv3d(channels[0], channels[1], 1),
                        nn.BatchNorm3d(channels[1]))
            else:
                self.shortcut = nn.Identity()

        #self.conv_in = nn.Conv2d(in_channel, channels[0], 3, stride=1, padding=1, bias=False)
        #self.bn_in = nn.BatchNorm2d(channels[0])
        #self.conv1 = nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=1)
        #self.bn1 = nn.BatchNorm2d(channels[1])
        #self.conv2 = nn.Conv2d(channels[1], channels[1], 3, stride=1, padding=1)
        #self.bn2 = nn.BatchNorm2d(channels[1])
        #if resblock:
        #    if channels[0]!=channels[1]:
        #        self.shortcut = nn.Sequential(
        #                nn.Conv2d(channels[0], channels[1], 1),
        #                nn.BatchNorm2d(channels[1]))
        #    else:
        #        self.shortcut = nn.Identity()

    def forward(self, x):
        '''
        x   : [n, h, w, l]
        out : [n, c, h, w, l]
        '''
        #n, h, w, l = x.shape
        #x = self.conv_in(x.view(n*h, w, l).unsqueeze(1))
        x = self.conv_in(x.unsqueeze(1))
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
        #_, c, w, l = out.shape
        #return torch.transpose(out.view(n, h, c, w, l), 1, 2)
        return out

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
            x  : [n, c, h, w, l]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)

class ConvHPP(nn.Module):
    def __init__(self, bin_num=[16], channel=512, window_size=[1,10,10]):
        feat_h = 16
        self.bin_num = bin_num
        super(ConvHPP, self).__init__()
        self.conv_mask = nn.ModuleDict({})
        for b in bin_num:
            window_size[0] = int(feat_h/b)
            self.conv_mask['bin{}'.format(b)] = nn.Sequential(
                    nn.Conv3d(channel, channel, kernel_size=window_size, stride=(window_size[0], 1, 1), bias=False),
                    nn.BatchNorm3d(channel),
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
            z = z.view(z.shape[:3]) + torch.max(x.view(n,c,b,-1), -1)[0]
            features.append(z)
        return torch.cat(features, -1)

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

class LossAggregator(nn.Module):
    def __init__(self, margin, scale, lamda) -> None:
        super().__init__()
        #self.TPloss = losses.TripletMarginLoss()
        #self.CEloss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.TPloss = TripletLoss(margin=margin, scale=scale, lamda=lamda, loss_term_weight=1)
        self.CEloss = CrossEntropyLoss(scale=16, loss_term_weight=0.1)

        self.TP_weight = 1#nn.Parameter(torch.Tensor(1))
        self.CE_weight = 0.1#nn.Parameter(torch.Tensor(1))
        self.miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="all")
        self.scale = 16

    def forward(self, TP_feat, CE_feat, labels):
        #TP_feat = TP_feat.view(TP_feat.shape[0], -1)
        #d = CE_feat.shape[-1]
        #
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

        loss_sum = TP_loss.mean()*self.TP_weight + CE_loss.mean()*self.CE_weight
        #end = time.time()
        #print('tp time: {}, ce time: {}, total: {}'.format(round(tp_end-st, 4), round(ce_end-tp_end, 4), round(end-st, 4)))
        return [loss_sum,TP_loss.mean(),CE_loss.mean()], mined_triplets

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
        dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)
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
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
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
        #index_ed = time.time()
        p, n, _ = dist.size()
        #matches_id = matches.nonzero(as_tuple=False)
        #diffenc_id = diffenc.nonzero(as_tuple=False)
        #ap_dist = dist[:, matches_id[:,0], matches_id[:,1]].view(p,n,-1,1)
        #an_dist = dist[:, diffenc_id[:,0], diffenc_id[:,1]].view(p,n,1,-1)
        ap_dist = dist[:, matches].view(p, n, -1, 1)
        an_dist = dist[:, diffenc].view(p, n, 1, -1)
        #ed = time.time()
        #print('map time: {}, extract time: {}'.format(round(index_ed-st, 4), round(ed-index_ed, 4)))
        return ap_dist, an_dist
