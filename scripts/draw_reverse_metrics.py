import numpy as np
import pandas as pd
import argparse
import torch
import os
import pickle
import plotly.express as px

def read_metrics(directory):
    metrics, idx_list = [], []
    for filename in os.listdir(directory):
        if filename not in ["Gs_G0.pkl", "Gs_Gt.pkl"]:
            file_path = os.path.join(directory, filename)
            idx = int(filename.split('.pkl')[0])
            idx_list.append(idx)

            # Read the .pkl file
            with open(file_path, 'rb') as file:
                metric = pickle.load(file)
            metrics.append(metric)
    return metrics, idx_list

def process_metrics_valid(metrics, e_st, e_end, e_int):
    metric_list = ['acc', 'prec', 'rec', 'ce']
    metrics_Gs = {'acc': [], 'prec': [], 'rec': [], 'ce': []}
    metrics_G0 = {'acc': [], 'prec': [], 'rec': [], 'ce': []}
    
    for metric_epoch in metrics:
        for metric_name in metric_list:
            metrics_Gs[metric_name].append(metric_epoch['Gs'][metric_name])
            metrics_G0[metric_name].append(metric_epoch['G0'][metric_name])
    
    num_steps = len(metrics_Gs['acc'][0])
    for metric_name in metric_list:
        metrics_Gs[metric_name] = torch.concat(metrics_Gs[metric_name]).numpy()
        metrics_G0[metric_name] = torch.concat(metrics_G0[metric_name]).numpy()
       
    _epoch_idx = list(range(9, e_end, e_int))
    _epoch_idx.insert(0, 0)
    epoch_idx = np.repeat(_epoch_idx, num_steps)

    step_idx = list(range(0, num_steps)) * len(_epoch_idx)

    dict_Gs = {'epoch_idx': epoch_idx, 'step_idx': step_idx}
    dict_Gs.update(metrics_Gs)

    dict_G0 = {'epoch_idx': epoch_idx, 'step_idx': step_idx}
    dict_G0.update(metrics_G0)

    df_Gs = pd.DataFrame(dict_Gs).melt(id_vars=['epoch_idx', 'step_idx'], value_vars=['acc', 'prec', 'rec', 'ce'], var_name='metric')
    df_G0 = pd.DataFrame(dict_G0).melt(id_vars=['epoch_idx', 'step_idx'], value_vars=['acc', 'prec', 'rec', 'ce'], var_name='metric')
    
    return df_Gs, df_G0

def process_metrics_test(metrics):
    metric_list = ['acc', 'prec', 'rec', 'ce']
    metrics_Gs = {'acc': None, 'prec': None, 'rec': None, 'ce': None}
    metrics_G0 = {'acc': None, 'prec': None, 'rec': None, 'ce': None}

    for metric_name in metric_list:
        metrics_Gs[metric_name] = metrics[0]['Gs'][metric_name].numpy() # we don't need the 0th metrics, useless
        metrics_G0[metric_name] = metrics[0]['G0'][metric_name].numpy()

    num_steps = len(metrics[0]['Gs']['acc'])

    step_idx = list(range(0, num_steps))

    dict_Gs = {'step_idx': step_idx}
    dict_Gs.update(metrics_Gs)

    dict_G0 = {'step_idx': step_idx}
    dict_G0.update(metrics_G0)

    df_Gs = pd.DataFrame(dict_Gs).melt(id_vars=['step_idx'], value_vars=['acc', 'prec', 'rec', 'ce'], var_name='metric')
    df_G0 = pd.DataFrame(dict_G0).melt(id_vars=['step_idx'], value_vars=['acc', 'prec', 'rec', 'ce'], var_name='metric')
    
    return df_Gs, df_G0

def process_metrics_infer(metrics):
    metric_list = ['acc', 'prec', 'rec', 'ce']
    metric_dict = {}

    for metric_name in metric_list:
        metric_dict[metric_name] = metrics[0][metric_name].numpy() # we don't need the 0th metrics, useless

    num_steps = len(metrics[0]['acc'])

    step_idx = list(range(0, num_steps))
    metric_dict.update({'step_idx': step_idx})

    df = pd.DataFrame(metric_dict).melt(id_vars=['step_idx'], value_vars=['acc', 'prec', 'rec', 'ce'], var_name='metric')
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', help="absolute dir path for chain results, e.g /home/jiahang/DiGress/chain_results/2024-04-13/11-28-34-protein-debug-2/protein-debug-2")
    parser.add_argument('--epoch_start', default=0, type=int)
    parser.add_argument('--epoch_end', default=20, type=int)
    parser.add_argument('--epoch_interval', default=10, type=int)
    parser.add_argument('--stages', nargs='+', default=['valid', 'test', 'infer'])
    parser.add_argument('--only_auroc', type=bool, default=False)

    args = parser.parse_args()

    root_path = args.path
    vis_path = os.path.join(root_path, 'vis')
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    for stage in args.stages:
        path = os.path.join(root_path, stage)
        metrics, idx_list = read_metrics(path)
        if stage == 'valid':
            df_Gs, df_G0 = process_metrics_valid(metrics, args.epoch_start, args.epoch_end, args.epoch_interval)
            fig_Gs = px.line(df_Gs, x="step_idx", y="value", color='epoch_idx', facet_col='metric')
            fig_G0 = px.line(df_G0, x="step_idx", y="value", color='epoch_idx', facet_col='metric')

            fig_Gs.write_html(os.path.join(root_path, 'vis', f'Gs_{stage}.html'))
            fig_G0.write_html(os.path.join(root_path, 'vis', f'G0_{stage}.html'))

            auroc_dict = {}
            for metric, idx in zip(metrics, idx_list):
                auroc_dict[idx] = f"{metric['G0']['auroc']:.4f}"
            print(f"{stage} G0 AUROC: {auroc_dict}")

        elif stage == 'test':
            df_Gs, df_G0 = process_metrics_test(metrics)
            fig_Gs = px.line(df_Gs, x="step_idx", y="value", facet_col='metric')
            fig_G0 = px.line(df_G0, x="step_idx", y="value", facet_col='metric')

            fig_Gs.write_html(os.path.join(root_path, 'vis', f'Gs_{stage}.html'))
            fig_G0.write_html(os.path.join(root_path, 'vis', f'G0_{stage}.html'))
            print(f"{stage} G0 AUROC: {metrics[0]['G0']['auroc']}")

        elif stage == 'infer':
            if not args.only_auroc:
                df = process_metrics_infer(metrics)
                fig = px.line(df, x="step_idx", y="value", facet_col='metric')
                fig.write_html(os.path.join(root_path, 'vis', f'G0_{stage}.html'))
            print(f"{stage} G0 AUROC: {metrics[0]['auroc']}")

        else:
            raise Exception(f"no stage {stage}")
        

    print("DONE!")

