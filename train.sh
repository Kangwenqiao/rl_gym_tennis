#!/usr/bin/env bash
# 若任何步骤报错则终止脚本
set -e

# 第一阶段：预训练 200 万步
python -u dqn_atari_more_reward_double.py \
    --exp-name test_stage1 \
    --gym-id TennisNoFrameskip-v4 \
    --learning-rate 0.0001 \
    --seed 2022 \
    --total-timesteps 2000000 \
    --learning-starts 100000 \
    --buffer-size 100000 \
    --save-interval 100000 \
    --batch-size 512 \
    --torch-deterministic true \
    --cuda true \
    --prod-mode false \
    --capture-video false \
    --eval-episode 3 \
    --gamma 0.99 \
    --target-network-frequency 1000 \
    --max-grad-norm 0.5 \
    --start-e 1 \
    --end-e 0.02 \
    --exploration-fraction 0.2 \
    --train-frequency 4 && \

# 第二阶段：接续训练
python -u dqn_atari_more_reward_double.py \
    --exp-name test_stage2 \
    --gym-id TennisNoFrameskip-v4 \
    --pretrained-model runs/test_stage1/model-2000000.pth \
    --learning-rate 0.00002 \
    --seed 2022 \
    --total-timesteps 10000000 \
    --learning-starts 100000 \
    --buffer-size 100000 \
    --save-interval 100000 \
    --batch-size 512 \
    --torch-deterministic true \
    --cuda true \
    --prod-mode false \
    --capture-video false \
    --eval-episode 3 \
    --gamma 0.99 \
    --target-network-frequency 1000 \
    --max-grad-norm 0.5 \
    --start-e 0.1 \
    --end-e 0.02 \
    --exploration-fraction 0.04 \
    --train-frequency 4 