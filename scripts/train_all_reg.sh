#!/bin/bash

DATA_PREFIX=../opennmt_preprocessed_data/iwslt-de-en

TRAIN_FROM=

UNIFORM_REG_LAMBDA=0.285
PERMUTE_REG_LAMBDA=0.142
ZERO_OUT_MAX_REG_LAMBDA=0.568

MODEL_RUN=1
EXPERIMENT_NAME=de-ft_from_r2_reward

MODEL_PATH=


LOG_FILE=../logs_winter/$EXPERIMENT_NAME
TENSORBOARD_LOG_FILE=../tensorboard_winter/$EXPERIMENT_NAME


CUDA_VISIBLE_DEVICES=0,1 python ../OpenNMT-py/train.py -world_size 2 -gpu_ranks 0 1 -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 1000 -valid_steps 1000 -train_steps 10000 -optim adam -learning_rate 0.001  -input_feed 0 -attn_reg --attn_reg_methods zero_out_max,random_permute,uniform  -zero_out_max_reg_lambda $ZERO_OUT_MAX_REG_LAMBDA -uniform_reg_lambda $UNIFORM_REG_LAMBDA -permute_reg_lambda $PERMUTE_REG_LAMBDA -train_from $TRAIN_FROM -reset_optim all # 1 gpu


