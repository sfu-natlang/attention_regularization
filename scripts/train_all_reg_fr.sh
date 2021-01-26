#!/bin/bash

DATA_PREFIX=../fr_data/iwslt-fr-en

#MODEL_PATH=/local-scratch/pooya/attention_regularization/models/reg1_lambda0
#LAMBDA_REG=0.02

#TRAIN_FROM=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/baseline/4/model_step_10000.pt
#TRAIN_FROM=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg_all/uni_0.33-perm_0.33-zo_max_0.33/1/model_step_15000.pt
TRAIN_FROM=/cs/natlang-expts/pooya/attention_regularization/pretrained/best_fr2.pt
#TRAIN_FROM=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/zom_1-R1-C2.5/model_step_30000.pt

UNIFORM_REG_LAMBDA=0
PERMUTE_REG_LAMBDA=0
ZERO_OUT_MAX_REG_LAMBDA=1

# EXPERIMENT_DIR=uni_$UNIFORM_REG_LAMBDA-perm_$PERMUTE_REG_LAMBDA-zo_max_$ZERO_OUT_MAX_REG_LAMBDA

#UNIFORM_REG_LAMBDA=0
#PERMUTE_REG_LAMBDA=0
#ZERO_OUT_MAX_REG_LAMBDA=1

MODEL_RUN=1
EXPERIMENT_NAME=fr_zom_R$MODEL_RUN-C2.5



#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg1/lambda$LAMBDA_REG/$MODEL_RUN/model
MODEL_PATH=/local-scratch/pooya/attention_regularization/fr-en/$EXPERIMENT_NAME/model

#MODEL_PATH=/local-scratch/pooya/attention_regularization/experiments/baseline/1/baseline1_ft
#TRAIN_FROM=/local-scratch/pooya/attention_regularization/experiments/baseline/1/baseline1_step_13000.pt

#LOG_FILE=../logs/baseline1

LOG_FILE=../logs/$EXPERIMENT_NAME
TENSORBOARD_LOG_FILE=../tensorboard/$EXPERIMENT_NAME



#python ../OpenNMT-py/train.py -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 5000 -valid_steps 500 -train_steps 50000 -optim adam -learning_rate 0.001 -log_file $LOG_FILE -input_feed 0 -world_size 2 -gpu_ranks 0 1 -tensorboard -tensorboard_log_dir ../tensorboard/baseline1 
#-attn_reg --attn_reg_methods second_max -lambda_reg 0

#python ../OpenNMT-py/train.py -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 1000 -valid_steps 500 -train_steps 50000 -log_file $LOG_FILE -input_feed 0 -world_size 2 -gpu_ranks 0 1 -tensorboard -tensorboard_log_dir ../tensorboard/baseline1-sgd0.5 -train_from $TRAIN_FROM -optim sgd -learning_rate 0.5 -reset_optim all

# 2 gpus
#python ../OpenNMT-py/train.py -world_size 2 -gpu_ranks 0 1 -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 1000 -valid_steps 500 -train_steps 15000 -optim adam -learning_rate 0.001  -input_feed 0 -attn_reg --attn_reg_methods zero_out_max,random_permute,uniform -uniform_reg_lambda $UNIFORM_REG_LAMBDA -permute_reg_lambda $PERMUTE_REG_LAMBDA -zero_out_max_reg_lambda $ZERO_OUT_MAX_REG_LAMBDA -tensorboard -tensorboard_log_dir $TENSORBOARD_LOG_FILE -log_file $LOG_FILE

CUDA_VISIBLE_DEVICES=0 python ../OpenNMT-py/train.py -world_size 1 -gpu_ranks 0 -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 5000 -valid_steps 1000 -train_steps 15000 -optim adam -learning_rate 0.001  -input_feed 0 -attn_reg --attn_reg_methods zero_out_max -zero_out_max_reg_lambda $ZERO_OUT_MAX_REG_LAMBDA -uniform_reg_lambda $UNIFORM_REG_LAMBDA -permute_reg_lambda $PERMUTE_REG_LAMBDA -tensorboard -tensorboard_log_dir $TENSORBOARD_LOG_FILE -log_file $LOG_FILE -train_from $TRAIN_FROM -reset_optim all
# 1 gpu
#CUDA_VISIBLE_DEVICES=1 python ../OpenNMT-py/train.py -world_size 1 -gpu_ranks 0 -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 1000 -valid_steps 500 -train_steps 15000 -optim adam -learning_rate 0.001  -input_feed 0 -attn_reg --attn_reg_methods zero_out_max,random_permute,uniform -uniform_reg_lambda $UNIFORM_REG_LAMBDA -permute_reg_lambda $PERMUTE_REG_LAMBDA -zero_out_max_reg_lambda $ZERO_OUT_MAX_REG_LAMBDA -tensorboard -tensorboard_log_dir $TENSORBOARD_LOG_FILE -log_file $LOG_FILE


