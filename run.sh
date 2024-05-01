#!/bin/bash

python3 src/main.py +experiment=protein dataset=protein general.name=protein-debug train.n_epochs=50

# python3 scripts/draw_reverse_metrics.py \
# --path /home/jiahang/DiGress/chain_results/2024-04-26/13-19-47-protein-debug/protein-debug \
# --stage valid test --epoch_end 19