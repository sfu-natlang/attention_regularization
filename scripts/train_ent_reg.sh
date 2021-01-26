#!/bin/bash

DATA_PREFIX=../opennmt_preprocessed_data/iwslt-de-en

MODEL_RUN=1
ENT_REG_LAMBDA=0.04
EXPERIMENT_NAME=ent-reg_run_$MODEL_RUN-lambda_$ENT_REG_LAMBDA


MODEL_PATH=/local-scratch/pooya/attention_regularization/$EXPERIMENT_NAME/model


LOG_FILE=../logs/$EXPERIMENT_NAME
TENSORBOARD_LOG_FILE=../tensorboard/$EXPERIMENT_NAME


# 1 gpus
 CUDA_AVAILABLE_DEVICES=0,1 python ../OpenNMT-py/train.py -world_size 2 -gpu_ranks 0 1 -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 5000 -valid_steps 500 -train_steps 30000 -optim adam -learning_rate 0.001  -input_feed 0 -ent_reg_lambda $ENT_REG_LAMBDA -tensorboard -tensorboard_log_dir $TENSORBOARD_LOG_FILE -log_file $LOG_FILE
