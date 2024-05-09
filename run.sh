#!/bin/bash

name="protein-4"
if [ "$1" == "train" ]; then
    # train
    echo 'train'
    python3 src/main.py +experiment=protein general.name=$name train.n_epochs=50 dataset.slope=50

elif [ "$1" == "infer" ]; then
    # infer
    echo 'infer'
    python3 src/main.py +experiment=protein general.name=$name general.infer=true \
    'general.test_only=/home/jiahang/DiGress/outputs/2024-05-03/02-13-00-protein-4/checkpoints/protein-4/epoch\=49.ckpt' \
    general.wandb=disabled dataset.slope=50

elif [ "$1" == "draw-valid-test" ]; then
    # draw lp metrics of reverse process
    echo 'draw-valid-test'
    if [ "$2" == "only_auroc" ]; then
        python3 scripts/draw_reverse_metrics.py \
        --path /home/jiahang/DiGress/chain_results/2024-05-02/07-54-50-protein-3/protein-3 \
        --stage valid test --epoch_end 20 --only_auroc

    else
        python3 scripts/draw_reverse_metrics.py \
        --path /home/jiahang/DiGress/chain_results/2024-05-02/07-54-50-protein-3/protein-3 \
        --stage valid test --epoch_end 20
    fi

elif [ "$1" == "draw-infer" ]; then
    echo 'draw-infer'
    addr="chain_results/2024-05-01/16-00-27-protein-2_infer/protein-2_infer"
    if [ "$2" == "only_auroc" ]; then
        python3 scripts/draw_reverse_metrics.py \
        --path $addr \
        --stage infer --epoch_end 20 --only_auroc
    else
        python3 scripts/draw_reverse_metrics.py \
        --path $addr \
        --stage infer --epoch_end 20
    fi

else
    echo "Unrecognized arg $1"
fi