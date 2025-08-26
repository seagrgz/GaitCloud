import os
import sys
import itertools
import datetime
import yaml
import random
import time
import numpy as np
import logging
import argparse
import subprocess
import json
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast
from pytorch_metric_learning import samplers

#customized modules
from util import config
from util.gait_database import GaitDataset
from util.collate_fn import CollateFn
import models
#from models.module import PCA_image

DEFAULT_ATTRIBUTES = ('memory.total','memory.free')

def collate_fn(batch):
    data = [sample[0] for sample in batch]
    label = torch.cat([sample[1] for sample in batch])
    return data, label

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def tensor_memory_size(tensor):
    # Number of elements in the tensor
    num_elements = tensor.numel()
    # Size of each element in bytes
    element_size = tensor.element_size()
    # Total memory consumed:
    # number of elements * size of each element
    memory_bytes = num_elements * element_size
    return memory_bytes

def model_memory_usage(model):
    # Calculate the model's parameter size in bytes
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    # Calculate the model's buffer size in bytes
    buffer_size = sum(p.numel() * p.element_size() for p in model.buffers())
    # Total memory occupied by the model's parameters and buffers
    total_size = param_size + buffer_size
    return total_size

def save_checkpoint(epoch_log, model, optimizer, scheduler, loss, args):
    filename = '[{}]/checkpoint[{}].pth'.format(args.timestamp, args.timestamp)
    logger.info('Saving checkpoint to: ' + filename)
    torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'losses':loss}, filename)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, help='config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg, args.config

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]
    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]

def main():
    args, configfile = get_parser()
    args.timestamp = str(datetime.datetime.now()).replace(" ", "_")
    os.system('mkdir [{}]'.format(args.timestamp))
    os.system('cp {} [{}]/config.yaml'.format(configfile, args.timestamp))
    with open(configfile, 'r') as f:
        configs = yaml.safe_load(f)

    if args.use_gpu and torch.cuda.is_available():
        #args.visible_gpu = list(range(torch.cuda.device_count()+1))
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    #data preparing
    if args.data_name in ['FreeGait', 'SUSTech1K']:
        if args.reload_data:
            print('Removing expired data...')
            os.system('rm -f /dev/shm/{}{}**'.format(args.identifier, args.data_name))
        if args.data_name == 'FreeGait':
            args.splits_variance = ['test']
            args.splits_view = []
        data_root = args.data_root
        datalist = {}

        if 'Norm' in args.structure:
            datalist['ref'] = [item for item in sorted(os.listdir(os.path.join(data_root, 'train'))) if '00-nm' in item]
        datalist['train'] = [item for item in sorted(os.listdir(os.path.join(data_root, 'train')))]
                #if ('dil8' in item) or ('dil' not in item)]
        datalist['test'] = [item for item in sorted(os.listdir(os.path.join(data_root, 'test')))]
        args.target = list(dict.fromkeys([int(item[:4]) for item in datalist['train']]))
    else:
        raise NotImplementedError('Dataset {} is not implemented!'.format(args.data_name))
    main_worker(args.train_gpu, args, data_root, datalist, configs)


def main_worker(gpu, args, data_root, datalist, configs):
    torch.autograd.set_detect_anomaly(True)
    if args.load_checkpoint:
        print('Loading checkpoint [{}]'.format(args.checkpoint_timestamp))
        checkpoint = torch.load('[{}]/checkpoint[{}].pth'.format(args.checkpoint_timestamp, args.checkpoint_timestamp), map_location=lambda storage, loc: storage.cuda())

    datasets = {}
    dataloaders = {}

    if args.use_gpu:
        device = torch.device('cuda:{}'.format(str(gpu[0])))
    else:
        device = torch.device('cpu')

    #data type init
    if args.datatype in ['double','float64']:
        args.dtype = torch.double
        args.use_bf16 = False
    elif args.datatype in ['float','float32']:
        args.dtype = torch.float
        args.use_bf16 = False
    elif args.datatype in ['half','bfloat16']:
        args.dtype = torch.float
        args.use_bf16 = True
    else:
        raise NotImplementedError('Invalid datatype or datatype name, got {}'.format(args.dtype))

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter()
    logger.info(args)
    model_name = 'runtime_state[{}].pth'.format(args.timestamp)

    #model init
    logger.info("=> creating model ...")
    Model = getattr(models, args.structure)
    model = Model(args)
    logger.info('network loaded')
    model.to(device).to(args.dtype)
    logger.info(model)
    logger.info('parameter number: {}'.format(count_parameters(model)))
    args.visual_name = {}

    #collate_fn init
    if args.structure == 'LidarGait':
        Collate_fn = CollateFn(frame_num=args.frame_size)
    else:
        Collate_fn = None

    if args.use_metric:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        Evaluator = getattr(models, 'MetricEvaluator')
        accuracy_calculator = Evaluator()
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.structure == 'LidarGait':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[27, 50], gamma=0.1)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.4*args.epochs), int(0.7*args.epochs)], gamma=0.1)

    #sampler init
    train_labels = [name[:4] for name in datalist['train']]
    train_sampler = samplers.MPerClassSampler(train_labels, args.pair_size, batch_size=args.batch_size, length_before_new_iter=len(datalist['train']))

    #dataset and dataloader init
    logger.info('Initializing datasets and dataloaders')
    #train
    datasets['train'] = GaitDataset(split='train', data_root=os.path.join(data_root, 'train'), args=args, datalist=datalist['train'])
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=args.batch_size, num_workers=args.workers, collate_fn=Collate_fn, sampler=train_sampler, pin_memory=True, drop_last=True)
    #test
    if args.eval_start < 1:
        datasets['test'] = GaitDataset(split='test', data_root=os.path.join(data_root, 'test'), args=args, datalist=datalist['test'])
        dataloaders['test'] = torch.utils.data.DataLoader(datasets['test'], batch_size=args.batch_size_test, num_workers=args.workers, collate_fn=Collate_fn, pin_memory=True, drop_last=False)
    if 'ref' in datalist.keys():
        datasets['ref'] = GaitDataset(split='ref', data_root=os.path.join(data_root, 'train'), args=args, datalist=datalist['ref'])
        dataloaders['ref'] = torch.utils.data.DataLoader(datasets['ref'], batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    if args.load_checkpoint:
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        loss_curve = checkpoint['loss']
    else:
        loss_curve = [[],[],[]]
    accuracy_val_curve = {}
    accuracy_train_curve = []
    best_view = 0
    best_var = 0

    logger.info('Train start ......')
    for epoch in range(args.start_epoch, args.epochs):
        epoch_st = time.time()
        accuracy_train, losses = metric_train(model, device, dataloaders['train'], optimizer, epoch, args, accuracy_calculator, scheduler)
        accuracy_train_curve.append(accuracy_train)
        for i in range(len(losses)):
            loss_curve[i] += losses[i]

        model_state = model.state_dict()
        train_end = time.time()

        #inference
        if epoch > args.epochs*args.eval_start or (args.visual and epoch%args.visual_freq == 0):
            print('use {} for test'.format(device))
            accuracy_val = metric_test(dataloaders['test'], model, accuracy_calculator, args, device, epoch)

            #update test accuracy
            for variance in args.splits_variance:
                if not variance in accuracy_val_curve.keys():
                    accuracy_val_curve[variance] = [accuracy_val[variance]]
                else:
                    accuracy_val_curve[variance].append(accuracy_val[variance])

            step_view = []
            for gallery in args.splits_view:
                if not gallery in accuracy_val_curve.keys():
                    accuracy_val_curve[gallery] = {}
                for variance in args.splits_view:
                    if not variance in accuracy_val_curve[gallery].keys():
                        accuracy_val_curve[gallery][variance] = [accuracy_val[gallery][variance]]
                    else:
                        accuracy_val_curve[gallery][variance].append(accuracy_val[gallery][variance])
                    step_view.append(accuracy_val[gallery][variance])

            step_var = [accuracy_val[attr] for attr in args.splits_variance]
            mean_var = 0
            mean_view = 0
            if not len(step_var) == 0:
                mean_var = sum(step_var)/len(step_var)
            if not len(step_view) == 0:
                mean_view = sum(step_view)/len(step_view)
            if mean_var > best_var:
                torch.save(model_state, '[{}]/best.pth'.format(args.timestamp))
                if mean_view > best_view and os.path.exists('[{}]/best_view.pth'.format(args.timestamp)):
                    os.system('rm -f [{}]/best_view.pth'.format(args.timestamp))
            elif mean_view > best_view:
                torch.save(model_state, '[{}]/best_view.pth'.format(args.timestamp))

        scheduler.step()
        epoch_end = time.time()
        logger.info('Train time: {}(s), Test time: {}(s), Total: {}(s)'.format(round(train_end-epoch_st, 3), round(epoch_end-train_end, 3), round(epoch_end-epoch_st, 3)))
        if args.save_checkpoint and epoch%args.check_freq == 0:
            logger.info('Saving checkpoint ...')
            save_checkpoint(epoch, model, optimizer, scheduler, loss_curve, args)

    #Train end
    if not os.path.exists('[{}]/best.pth'.format(args.timestamp)):
        torch.save(model_state, '[{}]/best.pth'.format(args.timestamp))
    writer.close()
    with open('[{}]/configfile[{}].yaml'.format(args.timestamp, args.timestamp), 'w') as f:
        configs['timestamp'] = args.timestamp
        yaml.dump(configs, f, allow_unicode=True, default_flow_style=False)
    os.system('rm -f /dev/shm/{}SUS**'.format(args.identifier))
    save_curve(accuracy_val_curve, accuracy_train_curve, loss_curve, args)
    logger.info('==>[{}]Training done!'.format(args.timestamp))

#metric learning
def metric_train(model, device, train_loader, optimizer, epoch, args, accuracy_calculator, scheduler):
    model.train()
    train_embeddings = []
    train_labels = []
    loss_items = [[],[],[]]
    kwargs = {}
    if args.structure in ['TestNorm','TestNorm_re']:
        refer_loader = dataloaders['ref']
        logger.info('Loading reference samples ...')
        ref_data = []
        ref_names = []
        for ref, _, metainfo in refer_loader:
            ref_data.append(ref.to(device).to(args.dtype))
            ref_names += metainfo[1]
        ref_data = torch.cat(ref_data)
        logger.info('Reference loaded')
    for batch_idx, (data, labels, metainfo) in enumerate(train_loader):
        loop_st = time.time()
        unique_labels = torch.unique(labels)
        if not len(labels) == len(unique_labels):
            data, labels, addons = data.to(device).to(args.dtype), labels.to(device).to(args.dtype), metainfo[0].to(device).to(args.dtype)

            #logger.info('Targets: {}, labels: {}'.format(metainfo[1], labels))
            if args.structure in ['TestNorm','TestNorm_re']:
                names = metainfo[1]
                target_names = []
                for i, name in enumerate(names):
                    attr, vp, target = name.split('_')
                    target_names.append('00-nm_' + vp + '_' + target)
                reference = []
                Id = []
                for i, tar in enumerate(target_names):
                    try:
                        reference.append(ref_data[ref_names.index(tar)])
                        Id.append(i)
                    except ValueError:
                        pass
                data = data[Id]
                labels = labels[Id]
                kwargs['ref'] = torch.stack(reference)

            optimizer.zero_grad()
            if args.use_bf16:
                batch_st = time.time()
                with autocast(dtype=torch.bfloat16):
                    (losses, mined_triplets, embeddings), visual_embed = model(data, labels=labels, training=True, addons=addons, visual=(args.visual and epoch%args.visual_freq == 0), **kwargs)
                    forward_end = time.time()
                    loss = losses[0]
                    logger.info("Epoch {} Iteration {}: Sum_loss = {}, TP_loss = {}, CE_loss = {}, Mined triplets = {}".format(epoch, batch_idx, loss.item(), losses[1].item(), losses[2].item(), mined_triplets))
                    #logger.info("Penalty: {}".format(losses[3].detach()))
                    loss.backward()
                    optimizer.step()
            else:
                raise NotImplementedError('Only support bfloat 16!')
            for i in range(len(loss_items)):
                loss_items[i].append(losses[i].item())
            backward_end = time.time()
            train_embeddings.append(embeddings.detach())
            train_labels.append(labels.detach())
            batch_end = time.time()
            logger.info('forward: {}, backward: {}, total: {}'.format(round(forward_end-batch_st, 4), round(backward_end-forward_end, 4), round(batch_end-batch_st, 4)))

            #visualization
            if args.visual and epoch%args.visual_freq == 0:
                for name in (set(metainfo[1]) & set(args.visual_train_names)):
                    visual_save(args.timestamp, epoch, name, visual_embed, metainfo[1].index(name))
        else:
            logger.info("Epoch {} Iteration {}: Mined triplets = 0, continue for next batch".format(epoch, batch_idx))
    if args.train_eval:
        train_embeddings = torch.cat(train_embeddings)
        train_labels = torch.cat(train_labels)
        accuracy, _ = accuracy_calculator.rank_1_accuracy(train_embeddings, train_labels, logger=logger)
        logger.info("Train set accuracy (Precision@1) = {}".format(accuracy))
    else:
        accuracy = 0.
    return accuracy, loss_items

def metric_test(test_loader, model, accuracy_calculator, args, device, epoch):
    model.eval().to(device)
    test_embeddings = []
    test_labels = []
    test_targets = []
    accuracies = {}
    test_num = {}
    
    #get embeddings
    for batch_idx, (data, labels, metainfo) in enumerate(test_loader):
        data, labels, addons = data.to(device).to(args.dtype), labels.to(device).to(args.dtype), metainfo[0].to(device).to(args.dtype)
        with torch.no_grad():
            if args.use_bf16:
                with autocast(dtype=torch.bfloat16):
                    embeddings, visual_embed = model(data, labels, training=False, addons=addons, visual=(args.visual and epoch%args.visual_freq == 0))
            else:
                embeddings, visual_embed = model(data, labels, training=False, addons=addons, visual=(args.visual and epoch%args.visual_freq == 0))
        test_embeddings.append(embeddings.detach().to('cpu'))
        test_labels.append(labels.detach().to('cpu'))
        test_targets += metainfo[1]

        if args.visual and epoch%args.visual_freq == 0:
            for name in (set(metainfo[1]) & set(args.visual_test_names)):
                visual_save(args.timestamp, epoch, name, visual_embed, metainfo[1].index(name))
    test_embeddings = torch.cat(test_embeddings)
    test_labels = torch.cat(test_labels)

    #variance / common
    var_targets = {}
    var_embeddings = {}
    var_labels = {}
    if args.data_name == 'SUSTech1K':
        for var in args.splits_variance:
            var_targets[var] = [item for item in test_targets if var in item]
            var_embeddings[var] = test_embeddings[[test_targets.index(item) for item in var_targets[var]]]
            var_labels[var] = test_labels[[test_targets.index(item) for item in var_targets[var]]]
            test_num[var] = len(var_labels[var])
    else:
        var_embeddings['test'] = test_embeddings
        var_labels['test'] = test_labels
        test_num['test'] = len(var_labels)
    for var in args.splits_variance:
        if var in ['00-nm', 'test']:
            accuracy, _ = accuracy_calculator.rank_1_accuracy(var_embeddings[var], var_labels[var], logger=logger)
        else:
            accuracy, _ = accuracy_calculator.rank_1_accuracy(var_embeddings[var], var_labels[var], var_embeddings['00-nm'], var_labels['00-nm'], logger)
        logger.info("{} samples in {} set, accuracy (R-1) = {}".format(test_num[var], var, accuracy))
        accuracies[var] = accuracy

    #view
    if args.data_name == 'SUSTech1K':
        view_targets = {}
        view_embeddings = {}
        view_labels = {}
        test_num = {}
        for vp in args.splits_view:
            view_targets[vp] = [item for item in test_targets if vp == item.split('_')[-1]]
            view_embeddings[vp] = test_embeddings[[test_targets.index(item) for item in view_targets[vp]]]
            view_labels[vp] = test_labels[[test_targets.index(item) for item in view_targets[vp]]]
            test_num[vp] = len(view_labels[vp])
        for gallery in args.splits_view:
            accuracies[gallery] = {}
            for probe in args.splits_view:
                if gallery == probe:
                    accuracy, _ = accuracy_calculator.rank_1_accuracy(view_embeddings[gallery], view_labels[gallery], logger=logger)
                else:
                    accuracy, _ = accuracy_calculator.rank_1_accuracy(view_embeddings[probe], view_labels[probe], view_embeddings[gallery], view_labels[gallery], logger)
                logger.info("{} to {} accuracy (Precision@1) = {}".format(gallery, probe, accuracy))
                accuracies[gallery][probe] = accuracy
        logger.info('view_after_eva-free memory: {}'.format(get_gpu_info()[0]['memory.free']))

    return accuracies

def visual_save(timestamp, epoch, name, embeds, idx):
    print('record {}'.format(name))
    if not os.path.exists('[{}]/{}'.format(timestamp, name)):
        os.system('mkdir [{}]/{}'.format(timestamp, name))
    for item in embeds.keys():
        #print(len(visual_embed[item]), len(metainfo[1]))
        np.save('[{}]/{}/{}_{}.npy'.format(timestamp, name, epoch, item), embeds[item][idx])

def save_curve(accuracy_val, accuracy_train, losses, args):
    assert len(accuracy_train) == int(args.epochs)
    axis_epoch = torch.arange(len(accuracy_train))
    try:
        axis_test = torch.arange(len(next(iter(accuracy_val.values()))))
        plot_accu = True
    except:
        plot_accu = False

    accu_record = np.array([accuracy_val, accuracy_train, losses], dtype=object)
    np.save('[{}]/final.npy'.format(args.timestamp), accu_record, allow_pickle=True)

    #train
    if not accuracy_train[-1] == 0:
        fig = plt.figure(figsize = (12, 6))
        plt.plot(axis_epoch, accuracy_train, label = 'accuracy_train')
        fig.legend()
        plt.title('best accuracy:{}'.format(max(accuracy_train)))
        plt.savefig('[{}]/train.png'.format(args.timestamp))
        plt.close()

    #save view
    if plot_accu:
        for gallery in args.splits_view:
            fig = plt.figure(figsize = (12,6)) 
            best_val = []
            for variance in args.splits_view:
                plt.plot(axis_test, accuracy_val[gallery][variance], label = 'accuracy_{}'.format(variance))
                fig.legend()
                best_val.append(max(accuracy_val[gallery][variance]))
            plt.title('best accuracy:{}'.format(best_val))
            plt.savefig('[{}]/{}.png'.format(args.timestamp, gallery))
            plt.close()
        #save variance
        for variance in args.splits_variance:
            fig = plt.figure(figsize = (12,6)) 
            plt.plot(axis_test, accuracy_val[variance], label = 'accuracy_{}'.format(variance))
            fig.legend()
            plt.title('best accuracy:{}'.format(max(accuracy_val[variance])))
            plt.savefig('[{}]/{}.png'.format(args.timestamp, variance))
            plt.close()

    #losses
    axis_iter = torch.arange(len(losses[0]))
    fig = plt.figure(figsize = (12, 6))
    plt.plot(axis_iter, losses[0], label = 'sum_loss')
    plt.plot(axis_iter, losses[1], label = 'Triplet_loss')
    plt.plot(axis_iter, losses[2], label = 'CEntropy_loss')
    fig.legend()
    plt.savefig('[{}]/losses.png'.format(args.timestamp))
    plt.close()

if __name__ == '__main__':
    import gc
    gc.collect()
    #print('main process start')
    main()
