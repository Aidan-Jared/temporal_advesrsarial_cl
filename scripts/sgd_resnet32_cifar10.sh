#!/bin/bash

# Exit on error
set -e

uv run main.py \
  --seed 42\
  --lr .1\
  --momentum 0.9\
  --batch_size 32\
  --task_epochs 2\
  --data_set "CIFAR10"\
  --task_splits 5\
  --dropout 0.0\
  --transform True\
  --task-shuffle False\
  --model "singleHeadResNet32"\
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]"\
  --method "SGD"\
  --poison_attacks '["gaussian_noise","shot_noise"]'\
  --poison_tasks "[0]"\
  --pcp .5\
  --pp .5
