#!/bin/bash

DATA_PREFIX=../opennmt_preprocessed_data/iwslt-de-en

#MODEL_PATH=/local-scratch/pooya/attention_regularization/models/reg1_lambda0

ZOM_UNK_LAMBDA=0.009375
MODEL_RUN=1
EXPERIMENT_NAME=zom_unk_$ZOM_UNK_LAMBDA-run_$MODEL_RUN


#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg1/lambda$LAMBDA_REG/$MODEL_RUN/model
MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/$EXPERIMENT_NAME/model

#MODEL_PATH=/local-scratch/pooya/attention_regularization/experiments/baseline/1/baseline1_ft
#TRAIN_FROM=/local-scratch/pooya/attention_regularization/experiments/baseline/1/baseline1_step_13000.pt

#LOG_FILE=../logs/baseline1



LOG_FILE=../logs/$EXPERIMENT_NAME
TENSORBOARD_LOG_FILE=../tensorboard2/$EXPERIMENT_NAME



#python ../OpenNMT-py/train.py -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 5000 -valid_steps 500 -train_steps 50000 -optim adam -learning_rate 0.001 -log_file $LOG_FILE -input_feed 0 -world_size 2 -gpu_ranks 0 1 -tensorboard -tensorboard_log_dir ../tensorboard/baseline1 
#-attn_reg --attn_reg_methods second_max -lambda_reg 0

#python ../OpenNMT-py/train.py -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 1000 -valid_steps 500 -train_steps 50000 -log_file $LOG_FILE -input_feed 0 -world_size 2 -gpu_ranks 0 1 -tensorboard -tensorboard_log_dir ../tensorboard/baseline1-sgd0.5 -train_from $TRAIN_FROM -optim sgd -learning_rate 0.5 -reset_optim all

# 2 gpus
#python ../OpenNMT-py/train.py -world_size 2 -gpu_ranks 0 1 -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 1000 -valid_steps 500 -train_steps 15000 -optim adam -learning_rate 0.001  -input_feed 0 --attn_reg_methods zero_out_max -zom_unk_lambda $ZOM_UNK_LAMBDA -tensorboard -tensorboard_log_dir $TENSORBOARD_LOG_FILE -log_file $LOG_FILE

#1 gpus
CUDA_VISIBLE_DEVICES=1  python ../OpenNMT-py/train.py -world_size 1 -gpu_ranks 0 -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 1000 -valid_steps 500 -train_steps 15000 -optim adam -learning_rate 0.001  -input_feed 0 --attn_reg_methods zero_out_max -zom_unk_lambda $ZOM_UNK_LAMBDA -tensorboard -tensorboard_log_dir $TENSORBOARD_LOG_FILE -log_file $LOG_FILE
