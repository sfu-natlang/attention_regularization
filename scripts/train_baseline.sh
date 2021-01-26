#!/bin/bash

DATA_PREFIX=

MODEL_PATH=

EXPERIMENT=de-baseline3
LOG_FILE=../logs_winter/$EXPERIMENT


# 2 gpus
CUDA_VISIBLE_DEVICES=0,1  python ../OpenNMT-py/train.py -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 5000 -valid_steps 1000 -train_steps 10000 -optim adam -learning_rate 0.001 -input_feed 0 -world_size 2 -gpu_ranks 0 1

