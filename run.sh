#!/bin/bash

# python3 src/main.py +experiment=protein dataset=protein general.name=protein-debug-4 train.n_epochs=50

python3 scripts/draw_reverse_metrics.py \
--path /home/jiahang/DiGress/chain_results/2024-04-25/05-45-36-protein-debug-4/protein-debug-4 \
--stage valid test --epoch_end 49