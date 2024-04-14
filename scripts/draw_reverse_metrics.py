import numpy as np
import pandas as pd
import argparse
import torch
import os
import pickle
import plotly.express as px

def read_metrics(stage, path):
    directory = path.format(stage=stage)
    metrics = []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            file_path = os.path.join(directory, filename)
            
            # Read the .pkl file
            with open(file_path, 'rb') as file:
                metric = pickle.load(file)
            metrics.append(metric)
    return metrics

def process_metrics_valid(metrics, e_st, e_end, e_int):
    metric_list = ['acc', 'prec', 'rec', 'ce']
    metrics_Gt_1_Gt = {'acc': [], 'prec': [], 'rec': [], 'ce': []}
    metrics_X_Gt = {'acc': [], 'prec': [], 'rec': [], 'ce': []}
    
    for metric_epoch in metrics:
        for metric_name in metric_list:
            metrics_Gt_1_Gt[metric_name].append(metric_epoch['Gt_1_Gt'][metric_name])
            metrics_X_Gt[metric_name].append(metric_epoch['X_Gt'][metric_name])
    
    num_steps = len(metrics_Gt_1_Gt['acc'][0])
    for metric_name in metric_list:
        metrics_Gt_1_Gt[metric_name] = torch.concat(metrics_Gt_1_Gt[metric_name][1:]).numpy() # we don't need the 0th metrics, useless
        metrics_X_Gt[metric_name] = torch.concat(metrics_X_Gt[metric_name][1:]).numpy()
       
    _epoch_idx = list(range(e_st, e_end + 1, e_int))
    epoch_idx = np.repeat(_epoch_idx, num_steps)

    step_idx = list(range(0, num_steps)) * len(_epoch_idx)

    dict_Gt_1_Gt = {'epoch_idx': epoch_idx, 'step_idx': step_idx}
    dict_Gt_1_Gt.update(metrics_Gt_1_Gt)

    dict_X_Gt = {'epoch_idx': epoch_idx, 'step_idx': step_idx}
    dict_X_Gt.update(metrics_X_Gt)

    df_Gt_1_Gt = pd.DataFrame(dict_Gt_1_Gt).melt(id_vars=['epoch_idx', 'step_idx'], value_vars=['acc', 'prec', 'rec', 'ce'], var_name='metric')
    df_X_Gt = pd.DataFrame(dict_X_Gt).melt(id_vars=['epoch_idx', 'step_idx'], value_vars=['acc', 'prec', 'rec', 'ce'], var_name='metric')
    
    return df_Gt_1_Gt, df_X_Gt

def process_metrics_test(metrics):
    metric_list = ['acc', 'prec', 'rec', 'ce']
    metrics_Gt_1_Gt = {'acc': None, 'prec': None, 'rec': None, 'ce': None}
    metrics_X_Gt = {'acc': None, 'prec': None, 'rec': None, 'ce': None}

    for metric_name in metric_list:
        metrics_Gt_1_Gt[metric_name] = metrics[0]['Gt_1_Gt'][metric_name].numpy() # we don't need the 0th metrics, useless
        metrics_X_Gt[metric_name] = metrics[0]['X_Gt'][metric_name].numpy()

    num_steps = len(metrics[0]['Gt_1_Gt']['acc'])

    step_idx = list(range(0, num_steps))

    dict_Gt_1_Gt = {'step_idx': step_idx}
    dict_Gt_1_Gt.update(metrics_Gt_1_Gt)

    dict_X_Gt = {'step_idx': step_idx}
    dict_X_Gt.update(metrics_X_Gt)

    df_Gt_1_Gt = pd.DataFrame(dict_Gt_1_Gt).melt(id_vars=['step_idx'], value_vars=['acc', 'prec', 'rec', 'ce'], var_name='metric')
    df_X_Gt = pd.DataFrame(dict_X_Gt).melt(id_vars=['step_idx'], value_vars=['acc', 'prec', 'rec', 'ce'], var_name='metric')
    
    return df_Gt_1_Gt, df_X_Gt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', help="absolute dir path for chain results, e.g /home/jiahang/DiGress/chain_results/2024-04-13/11-28-34-protein-debug-2/protein-debug-2")
    parser.add_argument('--save_every_epochs', default=10)
    parser.add_argument('--epoch_start', default=9)
    parser.add_argument('--epoch_end', default=99)
    parser.add_argument('--epoch_interval', default=10)
    parser.add_argument('--stages', nargs='+', default=['valid', 'test'])

    args = parser.parse_args()

    root_path = args.path
    vis_path = os.path.join(root_path, 'vis')
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    for stage in args.stages:
        path = os.path.join(root_path, stage)
        metrics = read_metrics('valid', path)
        if stage == 'valid':
            df_Gt_1_Gt, df_X_Gt = process_metrics_valid(metrics, args.epoch_start, args.epoch_end, args.epoch_interval)
            fig_Gt_1_Gt = px.line(df_Gt_1_Gt, x="step_idx", y="value", color='epoch_idx', facet_col='metric')
            fig_X_Gt = px.line(df_X_Gt, x="step_idx", y="value", color='epoch_idx', facet_col='metric')
        elif stage == 'test':
            df_Gt_1_Gt, df_X_Gt = process_metrics_test(metrics)
            fig_Gt_1_Gt = px.line(df_Gt_1_Gt, x="step_idx", y="value", facet_col='metric')
            fig_X_Gt = px.line(df_X_Gt, x="step_idx", y="value", facet_col='metric')
        else:
            raise Exception(f"no stage {stage}")
        
        fig_Gt_1_Gt.write_html(os.path.join(root_path, 'vis', f'Gt_1_Gt_{stage}.html'))
        fig_X_Gt.write_html(os.path.join(root_path, 'vis', f'X_Gt_{stage}.html'))

    print("DONE!")

