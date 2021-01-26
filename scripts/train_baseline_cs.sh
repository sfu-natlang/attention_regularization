#!/bin/bash

DATA_PREFIX=../cs_data/iwslt-cs-en

#MODEL_PATH=/local-scratch/pooya/attention_regularization/models/reg1_lambda0
#MODEL_PATH=/local-scratch/pooya/attention_regularization/experiments/baseline/1/model
#TRAIN_FROM=/local-scratch/pooya/attention_regularization/experiments/baseline/1/baseline1_step_13000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/experiments/iwslt.de-en/baseline/1/model
#MODEL_PATH=/local-scratch/pooya/attention_regularization/cs-en/baseline/3-1k/model
MODEL_PATH=/local-scratch/pooya/attention_regularization/aacl/cs/1/baseline/model

EXPERIMENT=cs-en-baseline3-1k
LOG_FILE=../logs/$EXPERIMENT

# 2 gpus
CUDA_VISIBLE_DEVICES=0,1 python ../OpenNMT-py/train.py -world_size 2 -gpu_ranks 0 1 -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 5000 -valid_steps 1000 -train_steps 10000 -optim adam -learning_rate 0.001 -input_feed 0
# 1 gpu
#CUDA_VISIBLE_DEVICES=1 python ../OpenNMT-py/train.py -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 5000 -valid_steps 1000 -train_steps 20000 -optim adam -learning_rate 0.001 -log_file $LOG_FILE -input_feed 0  -tensorboard -tensorboard_log_dir ../tensorboard/$EXPERIMENT -world_size 1 -gpu_ranks 0

#-attn_reg --attn_reg_methods second_max -lambda_reg 0

#python ../OpenNMT-py/train.py -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 1000 -valid_steps 500 -train_steps 50000 -log_file $LOG_FILE -input_feed 0 -world_size 2 -gpu_ranks 0 1 -tensorboard -tensorboard_log_dir ../tensorboard/baseline1-sgd0.5 -train_from $TRAIN_FROM -optim sgd -learning_rate 0.5 -reset_optim all

#python ../OpenNMT-py/train.py -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 5000 -valid_steps 500 -train_steps 50000 -optim adam -learning_rate 0.001  -input_feed 0 -attn_reg --attn_reg_methods second_max -lambda_reg  0
