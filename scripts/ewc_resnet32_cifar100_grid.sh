
#!/bin/bash
set -e

# Run 1/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 10 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 2/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 10 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 3/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 10 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 4/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 10 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 5/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 10 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 6/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 25 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 7/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 25 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 8/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 25 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 9/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 25 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 10/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 25 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 11/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 50 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 12/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 50 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 13/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 50 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 14/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 50 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 15/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 50 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 16/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 15 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 17/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 15 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 18/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 15 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 19/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 15 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 20/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 15 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 21/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 30 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 22/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 30 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 23/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 30 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 24/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 30 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 25/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 30 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 26/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 40 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 27/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 40 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 28/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 40 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 29/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 40 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 30/270
uv run main.py \
  --lr 0.01 \
  --batch_size 16 \
  --task_epochs 40 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 31/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 10 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 32/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 10 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 33/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 10 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 34/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 10 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 35/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 10 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 36/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 25 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 37/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 25 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 38/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 25 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 39/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 25 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 40/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 25 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 41/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 50 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 42/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 50 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 43/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 50 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 44/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 50 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 45/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 50 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 46/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 15 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 47/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 15 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 48/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 15 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 49/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 15 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 50/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 15 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 51/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 30 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 52/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 30 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 53/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 30 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 54/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 30 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 55/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 30 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 56/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 40 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 57/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 40 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 58/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 40 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 59/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 40 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 60/270
uv run main.py \
  --lr 0.01 \
  --batch_size 32 \
  --task_epochs 40 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 61/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 10 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 62/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 10 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 63/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 10 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 64/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 10 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 65/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 10 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 66/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 25 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 67/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 25 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 68/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 25 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 69/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 25 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 70/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 25 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 71/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 50 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 72/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 50 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 73/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 50 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 74/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 50 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 75/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 50 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 76/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 15 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 77/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 15 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 78/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 15 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 79/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 15 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 80/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 15 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 81/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 30 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 82/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 30 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 83/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 30 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 84/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 30 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 85/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 30 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 86/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 40 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 87/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 40 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 88/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 40 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 89/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 40 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 90/270
uv run main.py \
  --lr 0.01 \
  --batch_size 64 \
  --task_epochs 40 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 91/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 10 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 92/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 10 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 93/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 10 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 94/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 10 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 95/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 10 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 96/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 25 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 97/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 25 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 98/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 25 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 99/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 25 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 100/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 25 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 101/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 50 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 102/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 50 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 103/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 50 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 104/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 50 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 105/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 50 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 106/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 15 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 107/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 15 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 108/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 15 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 109/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 15 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 110/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 15 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 111/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 30 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 112/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 30 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 113/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 30 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 114/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 30 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 115/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 30 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 116/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 40 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 117/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 40 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 118/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 40 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 119/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 40 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 120/270
uv run main.py \
  --lr 0.05 \
  --batch_size 16 \
  --task_epochs 40 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 121/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 10 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 122/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 10 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 123/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 10 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 124/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 10 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 125/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 10 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 126/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 25 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 127/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 25 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 128/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 25 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 129/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 25 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 130/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 25 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 131/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 50 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 132/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 50 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 133/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 50 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 134/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 50 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 135/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 50 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 136/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 15 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 137/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 15 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 138/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 15 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 139/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 15 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 140/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 15 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 141/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 30 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 142/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 30 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 143/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 30 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 144/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 30 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 145/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 30 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 146/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 40 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 147/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 40 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 148/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 40 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 149/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 40 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 150/270
uv run main.py \
  --lr 0.05 \
  --batch_size 32 \
  --task_epochs 40 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 151/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 10 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 152/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 10 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 153/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 10 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 154/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 10 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 155/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 10 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 156/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 25 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 157/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 25 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 158/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 25 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 159/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 25 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 160/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 25 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 161/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 50 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 162/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 50 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 163/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 50 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 164/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 50 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 165/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 50 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 166/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 15 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 167/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 15 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 168/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 15 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 169/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 15 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 170/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 15 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 171/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 30 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 172/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 30 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 173/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 30 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 174/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 30 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 175/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 30 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 176/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 40 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 177/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 40 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 178/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 40 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 179/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 40 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 180/270
uv run main.py \
  --lr 0.05 \
  --batch_size 64 \
  --task_epochs 40 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 181/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 10 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 182/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 10 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 183/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 10 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 184/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 10 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 185/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 10 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 186/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 25 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 187/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 25 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 188/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 25 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 189/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 25 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 190/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 25 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 191/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 50 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 192/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 50 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 193/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 50 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 194/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 50 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 195/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 50 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 196/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 15 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 197/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 15 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 198/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 15 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 199/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 15 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 200/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 15 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 201/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 30 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 202/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 30 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 203/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 30 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 204/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 30 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 205/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 30 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 206/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 40 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 207/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 40 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 208/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 40 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 209/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 40 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 210/270
uv run main.py \
  --lr 0.1 \
  --batch_size 16 \
  --task_epochs 40 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 211/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 10 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 212/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 10 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 213/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 10 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 214/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 10 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 215/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 10 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 216/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 25 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 217/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 25 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 218/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 25 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 219/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 25 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 220/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 25 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 221/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 50 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 222/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 50 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 223/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 50 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 224/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 50 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 225/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 50 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 226/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 15 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 227/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 15 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 228/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 15 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 229/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 15 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 230/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 15 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 231/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 30 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 232/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 30 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 233/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 30 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 234/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 30 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 235/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 30 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 236/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 40 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 237/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 40 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 238/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 40 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 239/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 40 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 240/270
uv run main.py \
  --lr 0.1 \
  --batch_size 32 \
  --task_epochs 40 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 241/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 10 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 242/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 10 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 243/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 10 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 244/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 10 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 245/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 10 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 246/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 25 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 247/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 25 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 248/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 25 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 249/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 25 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 250/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 25 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 251/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 50 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 252/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 50 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 253/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 50 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 254/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 50 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 255/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 50 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 256/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 15 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 257/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 15 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 258/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 15 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 259/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 15 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 260/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 15 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 261/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 30 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 262/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 30 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 263/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 30 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 264/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 30 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 265/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 30 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 266/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 40 \
  --lambda_ 1e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 267/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 40 \
  --lambda_ 5e2 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 268/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 40 \
  --lambda_ 1e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 269/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 40 \
  --lambda_ 5e3 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

# Run 270/270
uv run main.py \
  --lr 0.1 \
  --batch_size 64 \
  --task_epochs 40 \
  --lambda_ 1e4 \
  --seed "[42]" \
  --data_set "CIFAR100" \
  --task_splits 5 \
  --model "multiHeadResNet32" \
  --norm "[(0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)]" \
  --method "EWC" \
  --poison_attacks '["gaussian_noise","shot_noise"]' \
  --poison_tasks "[0]" \
  --pcp .5 \
  --pp .5 

