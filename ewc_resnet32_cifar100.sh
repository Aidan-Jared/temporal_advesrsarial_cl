#!/bin/bash

# Exit on error
set -e

uv run main.py \
  --seed "[42]"\
  --lr .05\
  --batch_size 16\
  --task_epochs 25\
  --data_set "CIFAR100"\
  --task_splits 5\
  --model "multiHeadResNet32"\
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]"\
  --method "EWC"\
  --lambda_ 5e2\
  --poison_attacks '["gaussian_noise","shot_noise"]'\
  --poison_tasks "[0]"\
  --pcp .5\
  --pp .5
