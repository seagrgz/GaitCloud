#!/usr/bin/env python
import os
import sys
import yaml
import time
import numpy as np
import argparse
import json
import seaborn as sn
import matplotlib.pyplot as plt

import torch
from torch.cuda.amp import GradScaler, autocast

sys.path.append('./')
sys.path.append('util')
from util import config
from util.collate_fn import CollateFn
from util.gait_database import GaitDataset
import models

view_labels = ['000°', '045°', '090°', '135°', '180°', '225°', '270°', '315°', '*000°', '*090°', '*180°', '*270°']

def get_parser():
    parser = argparse.ArgumentParser(description='Load model parameters for inference')
    parser.add_argument('--time', type=str, help='training timestamp')
    parser.add_argument('--data', type=str, default=None, help='which data to use for inference')
    parser.add_argument('--visual', type=bool, default=False, help='if visualization')
    timestamp = parser.parse_args().time
    visual = parser.parse_args().visual
    data_root = parser.parse_args().data
    cfg = config.load_cfg_from_cfg_file('[{}]/config.yaml'.format(timestamp))
    cfg.target = list(dict.fromkeys([int(item[:4]) for item in os.listdir(cfg.data_root+'/train')]))
    if not data_root is None:
        cfg.data_root = data_root
    return cfg, timestamp, visual

def main():
    args, timestamp, visual = get_parser()
    print('config loaded')

    #common initialization
    args.timestamp = timestamp
    if args.structure == 'LidarGait':
        Collate_fn = CollateFn(frame_num=args.frame_size)
    else:
        Collate_fn = None
    args.symbol = None
    device = torch.device('cuda:0')
    args.dtype = torch.float
    Evaluator = getattr(models, 'MetricEvaluator')
    accuracy_calculator = Evaluator()

    print('loading {} in {}'.format(args.structure, timestamp))
    Model = getattr(models, args.structure)

    #load model
    model = Model(args)
    model.load_state_dict(torch.load('[{}]/best.pth'.format(timestamp)))
    model.eval().to(device)
    if os.path.exists('[{}]/best_view.pth'.format(args.timestamp)):
        model_view = Model(args)
        model_view.load_state_dict(torch.load('[{}]/best_view.pth'.format(timestamp)))
        model_view.eval().to(device)
    else:
        model_view = None
    print('network loaded')

    #for noise in [0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.1]:
    #print('set noise to {}'.format(noise))
    if visual:
        for group in args.visual_list.keys():
            v_set = GaitDataset(split='inference', data_root=os.path.join(args.data_root,group), args=args, datalist=args.visual_list[group], share_memory=False)
            v_loader = torch.utils.data.DataLoader(v_set, batch_size=8, num_workers=args.workers, collate_fn=Collate_fn, drop_last=False)
            for batch_idx, (data, labels, metainfo) in enumerate(v_loader):
                data, labels, addons = data.to(device).to(args.dtype), labels.to(device).to(args.dtype), metainfo[0].to(device).to(args.dtype)
                with autocast(dtype=torch.bfloat16):
                    _, visual_embed = model(data, labels, training=False, addons=addons, symbol=args.symbol, visual=True)
                for name in metainfo[1]:
                    visual_save(timestamp, 'best', name, visual_embed, metainfo[1].index(name))
    else:
        test_embeddings = {}
        test_labels = {}
        test_targets = {}
        data_root = os.path.join(args.data_root, 'test')
        data_list = [item for item in sorted(os.listdir(data_root))]
        if not len(data_list)==0:
            test_set = GaitDataset(split='test', data_root=data_root, args=args, datalist=data_list, share_memory=False)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size_test, num_workers=args.workers, collate_fn=Collate_fn, drop_last=False)

            test_embeddings['test'] = []
            test_labels['test'] = []
            test_targets['test'] = []

            #get embeddings
            for batch_idx, (data, labels, metainfo) in enumerate(test_loader):
                if batch_idx%10 == 0 and batch_idx != 0:
                    print('{} sample calculated'.format(batch_idx*args.batch_size_test))
                data, labels, addons = data.to(device).to(args.dtype), labels.to(device).to(args.dtype), metainfo[0].to(device).to(args.dtype)
                with torch.no_grad():
                    with autocast(dtype=torch.bfloat16):
                        embeddings, _ = model(data, labels, training=False, addons=addons, symbol=args.symbol)
                test_embeddings['test'].append(embeddings.detach().to('cpu'))
                test_labels['test'].append(labels.detach().to('cpu'))
                test_targets['test'] += metainfo[1]
            test_embeddings['test'] = torch.cat(test_embeddings['test'])
            test_labels['test'] = torch.cat(test_labels['test'])
            if not model_view is None: #best_variance model is not best_view
                for batch_idx, (data, labels, metainfo) in enumerate(test_loader):
                    if batch_idx%10 == 0 and batch_idx != 0:
                        print('{} sample calculated'.format(batch_idx*256))
                    data, labels, addons = data.to(device).to(args.dtype), labels.to(device).to(args.dtype), metainfo[0].to(device).to(args.dtype)
                    with torch.no_grad():
                        with autocast(dtype=torch.bfloat16):
                            embeddings, _ = model_view(data, labels, training=False, addons=addons, symbol=args.symbol)
                    test_embeddings['view'].append(embeddings.detach().to('cpu'))
                    test_labels['view'].append(labels.detach().to('cpu'))
                    test_targets['view'] += metainfo[1]
                test_embeddings['view'] = torch.cat(test_embeddings['view'])
                test_labels['view'] = torch.cat(test_labels['view'])
        else:
            test_embeddings['test'] = []
            test_labels['test'] = []
        print('embeddings calculation complete')

        if 'SUSTech1K' in args.data_root:
            splits_variance = ['00-nm', '01-nm', 'bg', 'cl', 'cr', 'ub', 'uf', 'oc', 'nt']
            splits_view = ['000', '045', '090', '135', '180', '225', '270', '315', '000-far', '090-near', '180-far', '270-far']
            #variance
            matrix_accu = []
            targets, embeddings, labels = test_targets['test'], test_embeddings['test'], test_labels['test']
            gallery_targets = [item for item in targets if '00-nm' in item]
            gallery_embeddings = embeddings[[targets.index(item) for item in gallery_targets]]
            gallery_labels = labels[[targets.index(item) for item in gallery_targets]]
            for probe in splits_variance:
                if not probe == '00-nm':
                    probe_targets = [item for item in targets if probe in item]
                    probe_embeddings = embeddings[[targets.index(item) for item in probe_targets]]
                    probe_labels = labels[[targets.index(item) for item in probe_targets]]
                    accuracy, _ = accuracy_calculator.rank_1_accuracy(probe_embeddings, probe_labels, gallery_embeddings, gallery_labels)
                    print('{} ({} samples): {}'.format(probe, len(probe_labels), accuracy))
            probe_targets = [item for item in targets if not '00-nm' in item]
            probe_embeddings = embeddings[[targets.index(item) for item in probe_targets]]
            probe_labels = labels[[targets.index(item) for item in probe_targets]]
            mean_var, _ = accuracy_calculator.rank_1_accuracy(probe_embeddings, probe_labels, gallery_embeddings, gallery_labels)
            print('Overall ({} samples): {}'.format(len(probe_labels), mean_var))

            #view
            view_accuracy = []
            view_dist = []
            if not model_view is None:
                targets, embeddings, labels = test_targets['view'], test_embeddings['view'], test_labels['view']
            for gallery in splits_view:
                g_accu = []
                gallery_targets = [item for item in targets if gallery == item.split('_')[-1]]
                gallery_embeddings = embeddings[[targets.index(item) for item in gallery_targets]]
                gallery_labels = labels[[targets.index(item) for item in gallery_targets]]
                for probe in splits_view:
                    probe_targets = [item for item in targets if probe == item.split('_')[-1]]
                    probe_embeddings = embeddings[[targets.index(item) for item in probe_targets]]
                    probe_labels = labels[[targets.index(item) for item in probe_targets]]
                    if not gallery == probe:
                        accuracy, _ = accuracy_calculator.rank_1_accuracy(probe_embeddings, probe_labels, gallery_embeddings, gallery_labels)
                        view_accuracy.append(accuracy)
                        view_dist.append(len(probe_targets))
                        g_accu.append(accuracy)
                    else:
                        accuracy, _ = accuracy_calculator.rank_1_accuracy(probe_embeddings, probe_labels)
                        g_accu.append(accuracy)
                matrix_accu.append(g_accu)
            mean_view = sum([view_accuracy[i]*view_dist[i] for i in range(len(view_dist))])/sum(view_dist)
            draw_matrix(timestamp, matrix_accu, mean_view)
            print('Overall cross_view accuracy: ', mean_view)
        
        if 'FreeGait' in args.data_root:
            targets, embeddings, labels = test_targets['test'], test_embeddings['test'], test_labels['test']
            with open('dataset/FreeGait/FreeGait_Data_Split.json', 'rb') as f:
                partition = json.load(f)
            f.close()
            probe_targets = partition['PROBE_SET']
            gallery_targets = [item for item in targets if not item in probe_targets]
            gallery_embeddings = embeddings[[targets.index(item) for item in gallery_targets]]
            gallery_labels = labels[[targets.index(item) for item in gallery_targets]]
            probe_embeddings = embeddings[[targets.index(item) for item in probe_targets]]
            probe_labels = labels[[targets.index(item) for item in probe_targets]]
            if len(probe_labels) == 0:
                if len(gallery_labels) == 0:
                    raise ValueError('Gallery should have at least 1 sample!')
                else:
                    print('Running self evaluation on gallery set...')
                    accuracy, _ = accuracy_calculator.rank_1_accuracy(gallery_embeddings, gallery_labels)
            else:
                accuracy, _ = accuracy_calculator.rank_1_accuracy(probe_embeddings, probe_labels, gallery_embeddings, gallery_labels)
            print('Overall accuracy ({} samples): {}'.format(len(probe_labels), accuracy))
        #np.save('{}/robustness.npy'.format(args.timestamp), np.array([mean_var, mean_view]))

def draw_matrix(timestamp, mx, overall):
    mx = (np.asarray(mx)*100).astype(int)
    vmin = 75 #min(mean_matrix_view.min(), max_matrix_view.min())
    vmax = 100 #max(mean_matrix_view.max(), max_matrix_view.max())
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=300)
    sn.heatmap(mx, annot=True, fmt='d', cmap='viridis', vmin=vmin, vmax=vmax, xticklabels=view_labels, yticklabels=view_labels)
    ax.set_title('Uniformed best accuracy ({})'.format(overall))
    ax.set_xlabel('probe')
    ax.set_ylabel('gallery')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('[{}]view_matrix.png'.format(timestamp))
    plt.close()

def visual_save(timestamp, epoch, name, embeds, idx):
    print('record {}'.format(name))
    if not os.path.exists('[{}]/{}'.format(timestamp, name)):
        os.system('mkdir [{}]/{}'.format(timestamp, name))
    for item in embeds.keys():
        #print(len(visual_embed[item]), len(metainfo[1]))
        np.save('[{}]/{}/{}_{}.npy'.format(timestamp, name, epoch, item), embeds[item][idx])

def fail_analyze(failed_info, probe_names, gallery_names):
    results = {}
    results_view = {}
    failed_names = [probe_names[i] for i in failed_info[0]]
    failed_preds = [gallery_names[i] for i in failed_info[1]]
    for name in probe_names:
        var, view, _ = name.split('_')
        if var in results.keys():
            if name in failed_names:
                results[var]['failed_name'][name] = failed_preds[failed_names.index(name)]
            results[var]['count'] += 1
        else:
            results[var] = {}
            if name in failed_names:
                results[var]['failed_name'] = {name:failed_preds[failed_names.index(name)]}
            else:
                results[var]['failed_name'] = {}
            results[var]['count'] = 1

        if view in results_view.keys():
            if name in failed_names:
                results_view[view]['failed_name'][name] = failed_preds[failed_names.index(name)]
            results_view[view]['count'] += 1
        else:
            results_view[view] = {}
            if name in failed_names:
                results_view[view]['failed_name'] = {name:failed_preds[failed_names.index(name)]}
            else:
                results_view[view]['failed_name'] = {}
            results_view[view]['count'] = 1

    for var in results.keys():
        results[var]['accuracy'] = (results[var]['count'] - len(results[var]['failed_name']))/results[var]['count']
    for view in results_view.keys():
        results_view[view]['accuracy'] = (results_view[view]['count'] - len(results_view[view]['failed_name']))/results_view[view]['count']
    return results, results_view

def draw_distribution(results, results_view, timestamp):
    for attr in results.keys():
        #variance
        x_labels = [name for name in results[attr].keys()]
        x_locs = np.arange(0, 5.9, 6/(len(x_labels)+1))[1:]
        width = 6/(len(x_labels)+1)/3
        accuracy = [results[attr][name]['accuracy'] for name in x_labels]
        count = [results[attr][name]['count'] for name in x_labels]
        fig, ax = plt.subplots(figsize=(6,4), dpi=300)
        ax.bar(x_locs-width/2, accuracy, width, label='Accuracy', color='tab:blue', zorder=2)
        addlabels(x_locs, width, accuracy)
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0,1)
        ax.set_xticks(x_locs, x_labels, fontsize=4)
        ax.set_zorder(1)
        ax.set_frame_on(False)
        ax_c = ax.twinx()
        ax_c.bar(x_locs+width/2, count, width, label='Count', color='tab:orange', zorder=2)
        ax_c.set_ylabel('Count')
        fig.legend()
        plt.savefig('[{}]/analysis/{}.png'.format(timestamp, attr))
        plt.close()

        #view
        x_labels = [name for name in results_view[attr].keys()]
        x_locs = np.arange(0, 5.9, 6/(len(x_labels)+1))[1:]
        width = 6/(len(x_labels)+1)/3
        accuracy = [results_view[attr][name]['accuracy'] for name in x_labels]
        count = [results_view[attr][name]['count'] for name in x_labels]
        fig, ax = plt.subplots(figsize=(6,4), dpi=300)
        ax.bar(x_locs-width/2, accuracy, width, label='Accuracy', color='tab:blue')
        ax.set_xticks(x_locs, x_labels, fontsize=5)
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0,1)
        ax_c = ax.twinx()
        ax_c.bar(x_locs+width/2, count, width, label='Count', color='tab:orange')
        ax_c.set_ylabel('Count')
        fig.legend()
        plt.savefig('[{}]/analysis/{}_view.png'.format(timestamp, attr))
        plt.close()

def addlabels(locs, width, y):
    for i in range(len(locs)):
        plt.text(locs[i]-width/2, y[i], round(y[i], 4), ha='center', fontsize=8, zorder=10)

def draw_robustness(var_accu, view_accu, timestamp):
    fig = plt.figure(figsize = (6,4),dpi=300) 
    noise = np.arange(0,0.105,0.005)
    noise = (3**0.5)*noise
    plt.plot(noise, np.asarray(var_accu), label = 'variance')
    plt.plot(noise, np.asarray(view_accu), label = 'view')
    fig.legend()
    plt.savefig('[{}]/robustness.png'.format(timestamp))
    plt.close()

if __name__ == '__main__':
    main()

    #timestamp = '[2024-10-11_07:43:50.842896]'
    #with open('{}/fail_analysis.yaml'.format(timestamp), 'r') as f:
    #    results = yaml.safe_load(f)
    #f.close()
    #with open('{}/fail_analysis_view.yaml'.format(timestamp), 'r') as f:
    #    results_view = yaml.safe_load(f)
    #f.close()
    #draw_distribution(results, results_view, timestamp)
