#!/bin/bash

# Exit on error
set -e

uv run main.py \
  --seed 42\
  --lr 1e-3\
  --batch_size 32\
  --task_epochs 50\
  --data_set "CIFAR10"\
  --task_splits 5\
  --model "multiHeadResNet32"\
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]"\
  --method "GEM"\
  --mem_size 200\
  --poison_attacks '["gaussian_noise","shot_noise"]'\
  --poison_tasks "[0]"\
  --pcp .5\
  --pp .5
