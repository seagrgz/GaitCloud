#!/usr/bin/env python
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from pytorch_metric_learning import losses, miners
from collections import OrderedDict
from ctypes import ArgumentError
import numpy as np

class LidarGait(nn.Module):
    def __init__(self, args, in_size):
        super().__init__()
        backbone = ResNet9(in_size=in_size)
        self.encoder = SetBlockWrapper(backbone)
        self.FCs = SeparateFCs()
        self.BNNecks = SeparateBNNecks(class_num=len(args.target))
        self.HPP = HorizontalPyramidPooling()
        self.TP = PackSequenceWrapper(torch.max, args.frame_size)
        self.mergeloss = LossAggregator()

    def forward(self, x, label=None, training=True, **kwargs):
        #ResNet9
        x = torch.transpose(x, 1, 2)
        #print('input size', x.shape)
        outs = self.encoder(x) #(n, c, s, h, w)
        #print('encoder out: ', out.shape)
        
        #Temporal pooling
        #print('TP input size', outs.shape)
        outs = self.TP(outs, options={"dim": 2})[0]

        #Horizontal Pyramid Pooling
        #print('HPP input size', outs.shape)
        feat = self.HPP(outs)
        #print('HPP out: ', feat.shape)

        #dense
        #print('feat size before FC', feat.shape)
        embed_tp = self.FCs(feat)
        #print('embeddings size ', embed_tp.shape)

        if training:
            #BNNeck for classification
            _, embed_ce = self.BNNecks(embed_tp)
            
            #print('embedding ce size', embed_ce.shape)
            #loss aggregation
            losses, mined_triplets = self.mergeloss(embed_tp, embed_ce, label)
            output = tuple([losses, mined_triplets, embed_tp])
        else:
            output = embed_tp

        return output

class ResNet9(ResNet):
    def __init__(self, in_size, maxpool=False):
        self.maxpool_flag = maxpool
        block = BasicBlock
        channels = [64,128,256,512]
        layers = [1,1,1,1]
        strides = [1,2,2,1]
        in_channels = in_size

        super(ResNet9, self).__init__(block, layers)

        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.conv1 = BasicConv2d(in_channels, self.inplanes, 3, 1, 1)

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool_flag:
           x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class HorizontalPyramidPooling():
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

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x

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

def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))

class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        """
            In  x: [n, c_in, s, h_in, w_in]
            Out x: [n, c_out, s, h_out, w_out]
        """
        n, c, s, h, w = x.size()
        x = self.forward_block(x.transpose(
            1, 2).reshape(-1, c, h, w), *args, **kwargs)
        output_size = x.size()
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()

class LossAggregator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #self.TPloss = losses.TripletMarginLoss()
        #self.CEloss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.TPloss = TripletLoss(margin=0.2, loss_term_weight=1)
        self.CEloss = CrossEntropyLoss(scale=16, loss_term_weight=0.1, log_accuracy=True)
        self.TP_weight = 1#nn.Parameter(torch.Tensor(1))
        self.CE_weight = 0.1#nn.Parameter(torch.Tensor(1))
        #self.miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="all")

        self.scale = 16

    def forward(self, TP_feat, CE_feat, labels):
        #TP_feat = TP_feat.view(TP_feat.shape[0], -1)
        #d = CE_feat.shape[-1]
        
        #indices_tuple = self.miner(TP_feat, labels)
        #TP_loss = self.TPloss(TP_feat, labels)
        #CE_loss = self.CEloss(CE_feat*self.scale, labels.unsqueeze(1).repeat(1, d).long())
        #print('TP_loss: {}/CE_loss: {}'.format(TP_loss, CE_loss))
        #mined_triplets = self.miner.num_triplets

        loss_sum = .0
        TP_loss, TP_loss_info = self.TPloss(TP_feat, labels)
        CE_loss, _ = self.CEloss(CE_feat, labels)
        mined_triplets = sum(TP_loss_info['loss_num'])

        loss_sum = TP_loss.mean()*self.TP_weight + CE_loss.mean()*self.CE_weight
        return [loss_sum,TP_loss.mean(),CE_loss.mean()], mined_triplets

def ddp_all_gather(features, dim=0, requires_grad=True):
    '''
        inputs: [n, ...]
    '''

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    feature_list = [torch.ones_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(feature_list, features.contiguous())

    if requires_grad:
        feature_list[rank] = features
    feature = torch.cat(feature_list, dim=dim)
    return feature

def gather_and_scale_wrapper(func):
    """Internal wrapper: gather the input from multple cards to one card, and scale the loss by the number of cards.
    """

    @functools.wraps(func)
    def inner(*args, **kwds):
        try:

            for k, v in kwds.items():
                kwds[k] = ddp_all_gather(v)

            loss, loss_info = func(*args, **kwds)
            loss *= torch.distributed.get_world_size()
            return loss, loss_info
        except:
            raise ArgumentError
    return inner

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
        self.info = Odict()

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
        return .0, self.info

class CrossEntropyLoss(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1, loss_term_weight=1.0, log_accuracy=False):
        super(CrossEntropyLoss, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy

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
        self.info.update({'loss': loss.detach().clone()})
        if self.log_accuracy:
            pred = logits.argmax(dim=1)  # [n, p]
            accu = (pred == labels).float().mean()
            self.info.update({'accuracy': accu})
        return loss, self.info

class TripletLoss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0):
        super(TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin

    #@gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [n, c, p], label: [n]
        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        mean_dist = dist.mean((1, 2))  # [p]
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
        dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)
        loss = F.relu(dist_diff + self.margin)

        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone()})

        return loss_avg, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

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
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        p, n, _ = dist.size()
        ap_dist = dist[:, matches].view(p, n, -1, 1)
        an_dist = dist[:, diffenc].view(p, n, 1, -1)
        return ap_dist, an_dist
