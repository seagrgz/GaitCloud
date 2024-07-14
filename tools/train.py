import os
import sys
import itertools
import datetime
import yaml
import random
import time
import nvtx
import numpy as np
import logging
import argparse
import shutil
import subprocess
import json
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from pytorch_metric_learning import distances, testers, samplers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from util import config
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.gait_database import GaitDataset
from util.collate_fn import CollateFn
import models

#import sentry_sdk

DEFAULT_ATTRIBUTES = ('memory.total','memory.free')
best_view = 0 #for updating overall accuracy on views

#sentry_sdk.init(
#    dsn="https://67519afdbbddda79fc3e71ea47b01cbc@o4506070819733504.ingest.sentry.io/4506070826942464",
#    # Set traces_sample_rate to 1.0 to capture 100%
#    # of transactions for performance monitoring.
#    traces_sample_rate=1.0,
#    # Set profiles_sample_rate to 1.0 to profile 100%
#    # of sampled transactions.
#    # We recommend adjusting this value in production.
#    profiles_sample_rate=1.0,
#)

def collate_fn(batch):
    data = [sample[0] for sample in batch]
    #print(batch[0][1])
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

def save_checkpoint(epoch_log, model, optimizer, scheduler, args):
    filename = '[{}]/checkpoint[{}].pth'.format(args.timestamp, args.timestamp)
    logger.info('Saving checkpoint to: ' + filename)
    torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}, filename)

def save_step_result(accuracy_val, accuracy_train, losses, args):
    accu_record = np.array([accuracy_val, accuracy_train, losses], dtype=object)
    np.save('[{}]/[{}].npy'.format(args.timestamp, args.timestamp), accu_record, allow_pickle=True)
    logger.info('Saving results to [{}].npy'.format(args.timestamp))

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/gait_database/gait_database_CNN_LSTM_repro.yaml', help='config file')
    parser.add_argument('opts', help='see config/gait_database/gait_database_CNN_LSTM_repro.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
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

def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)

def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]
    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]

def main():
    global gaitdata, gaitdata_test, label, label_test
    args, configfile = get_parser()
    args.timestamp = str(datetime.datetime.now()).replace(" ", "_")
    os.system('mkdir [{}]'.format(args.timestamp))
    print('Saving config file')
    with open(configfile, 'r') as f:
        configs = yaml.safe_load(f)

    if args.use_gpu and torch.cuda.is_available():
        #args.visible_gpu = list(range(torch.cuda.device_count()+1))
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    #if not args.test_split == 'test':
    #    args.epochs = 1000

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.data_name in ['gait_database', 'SUSTech1K']:
        if args.reload_data:
            os.system('rm -f /dev/shm/{}SUS**'.format(args.identifier))
        if args.data_name == 'gait_database':
            data_root = args.data_root.format(args.frame_size, args.data_split)
            data_list = [item[:-4] for item in sorted(os.listdir(data_root))]
        else:
            data_root = args.data_root
            splits = args.splits
            namespace = {}
        data_list = [name for name in [item[:-4] for item in sorted(os.listdir(data_root))] if not name == 'sample_list']
        args.target = [int(name) for name in np.load(os.path.join(data_root, 'train_list.npy'))]
        #args.target_test = [int(name) for name in np.load(os.path.join(data_root, 'test_list.npy'))]

        gallery = []
        probe = []

        probes = [item[:-4] for item in sorted(os.listdir(os.path.join(data_root, 'probe')))]
        for name in splits:
            if name == 'train':
                namespace[name] = [item[:-4] for item in sorted(os.listdir(os.path.join(data_root, name)))]
            else:
                namespace[name] = [item for item in probes if name in item]
    else:
        raise NotImplementedError()
    main_worker(args.train_gpu, args.ngpus_per_node, args, data_root, namespace, splits, configs)


def main_worker(gpu, ngpus_per_node, argss, data_root, namespace, splits, configs):
    global args, best_iou
    args = argss
    if args.load_checkpoint:
        print('Loading checkpoint [{}]'.format(args.checkpoint_timestamp))
        result = np.load('/home/sx-zhang/work/CNN-LSTM-master/[{}]/[{}].npy'.format(args.checkpoint_timestamp, args.checkpoint_timestamp), allow_pickle=True)
        checkpoint = torch.load('/home/sx-zhang/work/CNN-LSTM-master/[{}]/checkpoint[{}].pth'.format(args.checkpoint_timestamp, args.checkpoint_timestamp), map_location=lambda storage, loc: storage.cuda())
        accuracy_val_curve = result[0]
        accuracy_train_curve = result[1]
        loss_curve = result[2]
        failed_record = result[3]
    else:
        loss_curve = [[],[],[]]
        accuracy_val_curve = {}
        accuracy_train_curve = []
        failed_record = [0]

    #best_model_val = {}

    datasets = {}
    dataloaders = {}
    #print('mp_dis:{}, dis:{}'.format(args.multiprocessing_distributed, args.distributed))

    if args.use_gpu:
        device = torch.device('cuda:{}'.format(str(gpu[0])))
    else:
        device = torch.device('cpu')

    Collate_fn = None
    if args.structure == 'LidarGait':
        Collate_fn = CollateFn(frame_num=args.frame_size)
    in_size = args.feature
        
    #data type
    if args.datatype == 'double':
        args.dtype = torch.double
        args.use_bf16 = False
    elif args.datatype == 'float':
        args.dtype = torch.float
        args.use_bf16 = False
    elif args.datatype == 'half':
        args.dtype = torch.float
        args.use_bf16 = True
    else:
        raise NotImplementedError

    #model initialization
    Model = getattr(models, args.structure)
    model = Model(args, in_size)
    print('network loaded')
    #test_model = Model(args, in_size)
    model.to(device).to(args.dtype)
    print('parameter number:', count_parameters(model))

    if args.use_metric:
        #miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="all")
        #criterion = losses.TripletMarginLoss(margin=0.2)
        #optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, amsgrad=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        #accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
        Evaluator = getattr(models, 'MetricEvaluator')
        accuracy_calculator = Evaluator()
        scaler = GradScaler()
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        #optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, amsgrad=True)

    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.2), int(args.epochs*0.5), int(args.epochs*0.8)], gamma=0.1, verbose=True)
    if args.structure == 'LidarGait':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[27, 50], gamma=0.1, verbose=True)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70], gamma=0.1, verbose=True)

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info(model)
        model_name = 'runtime_state[{}].pth'.format(args.timestamp)

    #sampler initialization
    train_labels = [name[-4:] for name in namespace['train']]
    train_sampler = samplers.MPerClassSampler(train_labels, 2, batch_size=args.batch_size, length_before_new_iter=len(namespace['train']))

    #dataset and dataloader initialization
    logger.info('Initializing datasets and dataloaders')
    for name in splits:
        if name == 'train':
            datasets[name] = GaitDataset(split=name, data_root=os.path.join(data_root, name), args=args, datalist=namespace[name])
            dataloaders[name] = torch.utils.data.DataLoader(datasets[name], batch_size=args.batch_size, num_workers=args.workers, collate_fn=Collate_fn, sampler=train_sampler, pin_memory=True, drop_last=True)
        else:
            datasets[name] = GaitDataset(split=name, data_root=os.path.join(data_root, 'probe'), args=args, datalist=namespace[name])
            dataloaders[name] = torch.utils.data.DataLoader(datasets[name], batch_size=args.batch_size_test, num_workers=args.workers, collate_fn=Collate_fn, pin_memory=True, drop_last=False)

    if args.load_checkpoint:
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    for epoch in range(args.start_epoch, args.epochs):
        epoch_st = time.time()
        accuracy_train, losses = metric_train(model, device, dataloaders['train'], optimizer, epoch, args, accuracy_calculator, scheduler, scaler)
        accuracy_train_curve.append(accuracy_train)
        for i in range(len(losses)):
            loss_curve[i] += losses[i]

        #model_state = model.state_dict()
        train_end = time.time()

        #inference
        print('use {} for test'.format(device))
        accuracy_val, failed_targets = metric_test(dataloaders, model, accuracy_calculator, args, device, splits, epoch)

        #update test accuracy
        for variance in args.splits_variance:
            if epoch > 0:
                #if accuracy_val[variance] > max(accuracy_val_curve[variance]):
                #    best_model_val[variance] = model_state
                #    logger.info('Update best test model for {}'.format(variance))
                accuracy_val_curve[variance].append(accuracy_val[variance])
            else:
                accuracy_val_curve[variance] = [accuracy_val[variance]]
                accuracy_val_curve['overall'] = [0]
        for gallery in args.splits_view:
            for variance in args.splits_view:
                if epoch > 0:
                        accuracy_val_curve[gallery][variance].append(accuracy_val[gallery][variance])
                else:
                    if not gallery in accuracy_val_curve.keys():
                        accuracy_val_curve[gallery] = {}
                    accuracy_val_curve[gallery][variance] = [accuracy_val[gallery][variance]]
        if accuracy_val_curve['overall'][0] > 0:
            failed_record.append(failed_targets)
            if failed_record[0] == 0:
                failed_record[0] = epoch
        #logger.info(accuracy_val)
        scheduler.step()
        epoch_end = time.time()
        logger.info('Train time: {}(s), Test time: {}(s), Total: {}(s)'.format(round(train_end-epoch_st, 3), round(epoch_end-train_end, 3), round(epoch_end-epoch_st, 3)))
        save_step_result(accuracy_val_curve, accuracy_train_curve, loss_curve, args)

    if main_process():
        writer.close()
        with open('[{}]/configfile[{}]'.format(args.timestamp, args.timestamp), 'w') as f:
            configs['timestamp'] = args.timestamp
            yaml.dump(configs, f, allow_unicode=True, default_flow_style=False)
        os.system('rm -f /dev/shm/{}SUS**'.format(args.identifier))
        save_curve(accuracy_val_curve, accuracy_train_curve, loss_curve, failed_record, args, splits)
        logger.info('==>[{}]Training done!'.format(args.timestamp))

#metric learning
def metric_train(model, device, train_loader, optimizer, epoch, args, accuracy_calculator, scheduler, scaler):
    model.train()
    sum_embeddings = []
    sum_labels = []
    loss_items = [[],[],[]]
    for batch_idx, (data, labels, metainfo) in enumerate(train_loader):
        unique_labels = torch.unique(labels)
        #logger.info('start-free memory: {}'.format(get_gpu_info()[0]['memory.free']))
        if not len(labels) == len(unique_labels):
            data, labels, positions = data.to(device).to(args.dtype), labels.to(device).to(args.dtype), metainfo[0].to(device).to(args.dtype)
            optimizer.zero_grad()
            if args.use_bf16:
                batch_st = time.time()
                with autocast(dtype=torch.bfloat16):
                    losses, mined_triplets, embeddings = model(data, labels, positions=positions)
                    forward_end = time.time()
                    loss = losses[0]
                    logger.info("Epoch {} Iteration {}: Sum_loss = {}, TP_loss = {}, CE_loss = {}, Mined triplets = {}".format(epoch, batch_idx, loss.item(), losses[1].item(), losses[2].item(), mined_triplets))
                    loss.backward()
                    optimizer.step()
            else: #not used
                losses, mined_triplets, embeddings = model(data, labels)
                loss = losses[0]
                logger.info("Epoch {} Iteration {}: Sum_loss = {}, TP_loss = {}, CE_loss = {}, Mined triplets = {}".format(epoch, batch_idx, loss.item(), losses[1].item(), losses[2].item(), mined_triplets))
                loss.backward()
                optimizer.step()
            for i in range(len(loss_items)):
                loss_items[i].append(losses[i].item())
            #sum_embeddings[data_count:data_count+len(labels)] = embeddings.view(embeddings.shape[0], -1).detach()
            backward_end = time.time()
            sum_embeddings.append(embeddings.detach())
            sum_labels.append(labels.detach())
            batch_end = time.time()
            logger.info('forward: {}, backward: {}, total: {}'.format(round(forward_end-batch_st, 4), round(backward_end-forward_end, 4), round(batch_end-batch_st, 4)))
        else:
            logger.info("Epoch {} Iteration {}: Mined triplets = 0, continue for next batch".format(epoch, batch_idx))
    sum_embeddings = torch.cat(sum_embeddings)
    sum_labels = torch.cat(sum_labels)
    #GPU_info = get_gpu_info()
    #if int(GPU_info[0]['memory.free']) < args.free_memory_required:
    #    save_checkpoint(epoch, model, optimizer, scheduler, args)

    #accuracy = accuracy_calculator.get_accuracy(sum_embeddings, sum_labels, sum_embeddings, sum_labels, True)["precision_at_1"]
    accuracy, _ = accuracy_calculator.rank_1_accuracy(sum_embeddings, sum_labels, logger=logger)
    del sum_embeddings, sum_labels, losses
    logger.info("Train set accuracy (Precision@1) = {}".format(accuracy))
    return accuracy, loss_items

def metric_test(dataloaders, model, accuracy_calculator, args, device, splits, epoch):
    global best_view
    model.eval().to(device)
    #gallery_embeddings = torch.zeros((len(dataloaders['00-nm'])*args.batch_size_test,256*16), device=device)
    gallery_embeddings = []
    gallery_labels = []

    variance_embeddings = {}
    variance_labels = {}
    variance_targets = {}
    variance_failed = {}

    accuracies = {}
    
    #variance
    variance_targets['00-nm'] = []
    for batch_idx, (data, labels, metainfo) in enumerate(dataloaders['00-nm']):
        data, labels, positions = data.to(device).to(args.dtype), labels.to(device).to(args.dtype), metainfo[0].to(device).to(args.dtype)
        with torch.no_grad():
            if args.use_bf16:
                with autocast(dtype=torch.bfloat16):
                    embeddings = model(data, labels, training=False, positions=positions)
            else:
                embeddings = model(data, labels, training=False, positions=positions)
        #gallery_embeddings[data_count:data_count+len(labels)] = embeddings.view(embeddings.shape[0], -1).detach()
        gallery_embeddings.append(embeddings.detach())
        gallery_labels.append(labels.detach())
        variance_targets['00-nm'] += metainfo[1]
    variance_embeddings['00-nm'] = torch.cat(gallery_embeddings)
    variance_labels['00-nm'] = torch.cat(gallery_labels)
    del gallery_embeddings, gallery_labels

    #variance_targets['00-nm'] 

    for variance in args.splits_variance:
        if not variance == '00-nm':
            t_embeddings = []
            t_labels = []
            variance_targets[variance] = []
            for batch_idx, (data, labels, metainfo) in enumerate(dataloaders[variance]):
                data, labels, positions = data.to(device).to(args.dtype), labels.to(device).to(args.dtype), metainfo[0].to(device).to(args.dtype)
                with torch.no_grad():
                    if args.use_bf16:
                        with autocast(dtype=torch.bfloat16):
                            embeddings = model(data, labels, training=False, positions=positions)
                    else:
                        embeddings = model(data, labels, training=False, positions=positions)
                #t_embeddings[data_count:data_count+len(labels)] = embeddings.view(embeddings.shape[0], -1).detach()
                t_embeddings.append(embeddings.detach())
                t_labels.append(labels.detach())
                variance_targets[variance] += metainfo[1]
            variance_embeddings[variance] = torch.cat(t_embeddings)
            variance_labels[variance] = torch.cat(t_labels)
            #logger.info('evaluation-free memory: {}'.format(get_gpu_info()[0]['memory.free']))

            #accuracy = accuracy_calculator.get_accuracy(t_embeddings, t_labels, gallery_embeddings, gallery_labels, False)["precision_at_1"]
            accuracy, failed_indices = accuracy_calculator.rank_1_accuracy(variance_embeddings[variance], variance_labels[variance], variance_embeddings['00-nm'], variance_labels['00-nm'], logger)
        else:
            #accuracy = accuracy_calculator.get_accuracy(gallery_embeddings, gallery_labels, gallery_embeddings, gallery_labels, True)["precision_at_1"]
            accuracy, failed_indices = accuracy_calculator.rank_1_accuracy(variance_embeddings[variance], variance_labels[variance], logger=logger)
        #print(gallery_embeddings.shape, gallery_labels.shape, t_embeddings.shape, t_labels.shape)
        logger.info("{} set accuracy (Precision@1) = {}".format(variance, accuracy))
        accuracies[variance] = accuracy
        variance_failed[variance] = [variance_targets[variance][Id] for Id in failed_indices]
    test_names = list(itertools.chain(*variance_targets.values()))
    test_embeddings = torch.cat([*variance_embeddings.values()])
    test_labels = torch.cat([*variance_labels.values()])
    #logger.info('var_end-free memory: {}'.format(get_gpu_info()[0]['memory.free']))

    #view
    view_embeddings = {}
    view_labels = {}
    view_targets = {}
    view_failed = {}
    for vp in args.splits_view:
        view_targets[vp] = list(dict.fromkeys([item for item in test_names if vp in item]))
        view_embeddings[vp] = test_embeddings[[test_names.index(item) for item in view_targets[vp]]]
        view_labels[vp] = test_labels[[test_names.index(item) for item in view_targets[vp]]]
        #print(view_embeddings[vp].shape, view_labels[vp].shape, len(view_targets[vp]))

    #for name in args.splits_view:
    #    #t_embeddings = torch.zeros((len(dataloaders[name])*args.batch_size_test,256*16), device=device)
    #    t_embeddings = []
    #    t_labels = []
    #    view_targets[name] = []
    #    for batch_idx, (data, labels, metainfo) in enumerate(dataloaders[name]):
    #        data, labels, positions = data.to(device).to(args.dtype), labels.to(device).to(args.dtype), metainfo[0].to(device).to(args.dtype)
    #        with torch.no_grad():
    #            if args.use_bf16:
    #                with autocast(dtype=torch.bfloat16):
    #                    embeddings = model(data, labels, training=False, positions=positions)
    #            else:
    #                embeddings = model(data, labels, training=False, positions=positions)
    #        #t_embeddings[data_count:data_count+len(labels)] = embeddings.view(embeddings.shape[0], -1).detach()
    #        t_embeddings.append(embeddings.detach())
    #        t_labels.append(labels.detach())
    #        view_targets[name] += metainfo[1]
    #    view_embeddings[name] = torch.cat(t_embeddings)
    #    view_labels[name] = torch.cat(t_labels)
    #    #logger.info('evaluation-free memory: {}'.format(get_gpu_info()[0]['memory.free']))
    #    if len(view_labels[name].shape) == 2:
    #        view_labels[name] = view_labels[name].squeeze(1)
    ##logger.info('view_before_eva-free memory: {}'.format(get_gpu_info()[0]['memory.free']))

    step_view = []
    for gallery in args.splits_view:
        accuracies[gallery] = {}
        view_failed[gallery] = {}
        for probe in args.splits_view:
            if gallery == probe:
                #accuracy = accuracy_calculator.get_accuracy(view_embeddings[gallery], view_labels[gallery], view_embeddings[gallery], view_labels[gallery], True)["precision_at_1"]
                accuracy, failed_indices = accuracy_calculator.rank_1_accuracy(view_embeddings[gallery], view_labels[gallery], logger=logger)
            else:
                #accuracy = accuracy_calculator.get_accuracy(view_embeddings[probe], view_labels[probe], view_embeddings[gallery], view_labels[gallery], False)["precision_at_1"]
                accuracy, failed_indices = accuracy_calculator.rank_1_accuracy(view_embeddings[probe], view_labels[probe], view_embeddings[gallery], view_labels[gallery], logger)
            logger.info("{} to {} accuracy (Precision@1) = {}".format(gallery, probe, accuracy))
            accuracies[gallery][probe] = accuracy
            step_view.append(accuracy)
            view_failed[gallery][probe] = [view_targets[probe][Id] for Id in failed_indices]
    logger.info('view_after_eva-free memory: {}'.format(get_gpu_info()[0]['memory.free']))
    mean_step = sum(step_view)/len(step_view)
    
    #overall
    if mean_step > best_view and mean_step > 0.9 and epoch > 1*(args.epochs): #set min of mean accuracy for time saving
        view_embeddings = torch.cat([view_embeddings[name] for name in args.splits_view])
        view_labels = torch.cat([view_labels[name] for name in args.splits_view])
        overall, _ = accuracy_calculator.rank_1_accuracy(view_embeddings, view_labels, logger=logger)
        accuracies['overall'] = [overall, epoch]
    return accuracies, (variance_failed, view_failed)

def save_curve(accuracy_val, accuracy_train, losses, failed, args, splits):
    assert len(accuracy_train) == int(args.epochs)
    axis_epoch = torch.arange(len(accuracy_train))

    #conver_e = 0
    #step_accu = []
    #logger.info(accuracy_val)
    #for step in range(args.mean_range,args.epochs):
    #    st_mean = np.mean(accuracy_val[step-args.mean_range:step]) 
    #    step_accu.append(st_mean)
    #best_accu = max(step_accu)
    #best_pos = step_accu.index(best_accu)
    #best_var = np.var(accuracy_val[best_pos:best_pos+args.mean_range])

    ##calculate convergence time
    #for pos, step_accuracy in enumerate(step_accu):
    #    step_var = np.var(accuracy_val[pos:pos+args.mean_range])
    #    if step_accuracy > best_accu-0.2 and step_var < best_var+0.0001:
    #        conver_e = pos+args.mean_range
    #        break
    accu_record = np.array([accuracy_val, accuracy_train, losses, failed], dtype=object)

    #train
    fig = plt.figure(figsize = (12, 6))
    plt.plot(axis_epoch, accuracy_train, label = 'accuracy_train')
    fig.legend()
    plt.title('best accuracy:{}'.format(max(accuracy_train)))
    plt.savefig('[{}]/train.png'.format(args.timestamp))
    #torch.save(best_model_train, '{}_{}+{}_{}_lr{}+{}[{}]train.pth'.format(args.datatype, args.data_split, args.feature, args.structure, args.base_lr, args.epochs, args.timestamp))
    np.save('[{}]/final.npy'.format(args.timestamp), accu_record, allow_pickle=True)
    plt.close()

    #save view
    for gallery in args.splits_view:
        fig = plt.figure(figsize = (12,6)) 
        best_val = []
        for variance in args.splits_view:
            plt.plot(axis_epoch, accuracy_val[gallery][variance], label = 'accuracy_{}'.format(variance))
            fig.legend()
            best_val.append(max(accuracy_val[gallery][variance]))
        plt.title('best accuracy:{}'.format(best_val))
        plt.savefig('[{}]/{}.png'.format(args.timestamp, gallery))
        plt.close()
    #save variance
    for variance in args.splits_variance:
        fig = plt.figure(figsize = (12,6)) 
        plt.plot(axis_epoch, accuracy_val[variance], label = 'accuracy_{}'.format(variance))
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
    #mp.set_start_method('spawn')
    gc.collect()
    #print('main process start')
    main()
