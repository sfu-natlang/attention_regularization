#!/bin/bash

DATA_PREFIX=../cs_data/iwslt-cs-en

#MODEL_PATH=/local-scratch/pooya/attention_regularization/models/reg1_lambda0
#LAMBDA_REG=0.02

#TRAIN_FROM=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/baseline/4/model_step_10000.pt
#TRAIN_FROM=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg_all/uni_0.33-perm_0.33-zo_max_0.33/1/model_step_15000.pt
#TRAIN_FROM=/cs/natlang-expts/pooya/attention_regularization/pretrained/best_cs.pt
#TRAIN_FROM=/cs/natlang-expts/pooya/attention_regularization/pretrained/cs_best3_for_ft.pt

#TRAIN_FROM=/local-scratch/pooya/attention_regularization/cs-en/baseline/3-1k/model_step_10000.pt
TRAIN_FROM=/local-scratch/pooya/attention_regularization/aacl/cs/1/baseline/model_step_10000.pt

#TRAIN_FROM=/local-scratch/pooya/attention_regularization/cs-en/ft/model_step_5000.pt #continue until 10 
#TRAIN_FROM=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/zom_1-R1-C2.5/model_step_30000.pt

UNIFORM_REG_LAMBDA=0.285
PERMUTE_REG_LAMBDA=0.142
ZERO_OUT_MAX_REG_LAMBDA=0.568

# EXPERIMENT_DIR=uni_$UNIFORM_REG_LAMBDA-perm_$PERMUTE_REG_LAMBDA-zo_max_$ZERO_OUT_MAX_REG_LAMBDA

#UNIFORM_REG_LAMBDA=0
#PERMUTE_REG_LAMBDA=0
#ZERO_OUT_MAX_REG_LAMBDA=1

EXPERIMENT_NAME=ft2



#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg1/lambda$LAMBDA_REG/$MODEL_RUN/model
MODEL_PATH=/local-scratch/pooya/attention_regularization/aacl/cs/1/ft_from_baseline/model

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

CUDA_VISIBLE_DEVICES=0 python ../OpenNMT-py/train.py -world_size 1 -gpu_ranks 0 -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 1000 -valid_steps 1000 -train_steps 10000 -optim adam -learning_rate 0.001  -input_feed 0 -attn_reg --attn_reg_methods zero_out_max,random_permute,uniform  -zero_out_max_reg_lambda $ZERO_OUT_MAX_REG_LAMBDA -uniform_reg_lambda $UNIFORM_REG_LAMBDA -permute_reg_lambda $PERMUTE_REG_LAMBDA -train_from $TRAIN_FROM -reset_optim all # 1 gpu
#CUDA_VISIBLE_DEVICES=1 python ../OpenNMT-py/train.py -world_size 1 -gpu_ranks 0 -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 1000 -valid_steps 500 -train_steps 15000 -optim adam -learning_rate 0.001  -input_feed 0 -attn_reg --attn_reg_methods zero_out_max,random_permute,uniform -uniform_reg_lambda $UNIFORM_REG_LAMBDA -permute_reg_lambda $PERMUTE_REG_LAMBDA -zero_out_max_reg_lambda $ZERO_OUT_MAX_REG_LAMBDA -tensorboard -tensorboard_log_dir $TENSORBOARD_LOG_FILE -log_file $LOG_FILE

