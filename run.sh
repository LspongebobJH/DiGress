#!/bin/bash

name="protein-2"
if [ "$1" == "train" ]; then
    # train
    echo 'train'
    python3 src/main.py +experiment=protein general.name=$name train.n_epochs=20 dataset.slope=50

elif [ "$1" == "infer" ]; then
    # infer
    echo 'infer'
    python3 src/main.py +experiment=protein general.name=$name general.infer=true \
    'general.test_only=/home/jiahang/DiGress/outputs/2024-05-01/07-25-48-protein-1/checkpoints/protein-1/epoch\=9.ckpt' \
    general.wandb=disabled

elif [ "$1" == "draw" ]; then
    # draw lp metrics of reverse process
    echo 'draw'
    python3 scripts/draw_reverse_metrics.py \
    --path /home/jiahang/DiGress/chain_results/2024-05-01/16-00-27-protein-2/protein-2 \
    --stage valid test --epoch_end 20

    # python3 scripts/draw_reverse_metrics.py \
    # --path /home/jiahang/DiGress/chain_results/2024-05-01/07-25-48-protein-1_resume/protein-1_resume \
    # --stage infer --epoch_end 20

else
    echo "Unrecognized arg $1"
fi