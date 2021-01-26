#!/bin/bash

DATA_PREFIX=../opennmt_preprocessed_data/iwslt-de-en

UNIFORM_REG_LAMBDA=0.285
PERMUTE_REG_LAMBDA=0.142
ZERO_OUT_MAX_REG_LAMBDA=0.568

MODEL_PATH=/local-scratch/pooya/attention_regularization/$EXPERIMENT_NAME/model


python ../OpenNMT-py/train.py -encoder_type brnn -global_attention general -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 5000 -valid_steps 1000 -train_steps 30000 -optim adam -learning_rate 0.001  -input_feed 0 -attn_reg --attn_reg_methods zero_out_max,random_permute,uniform  -zero_out_max_reg_lambda $ZERO_OUT_MAX_REG_LAMBDA -uniform_reg_lambda $UNIFORM_REG_LAMBDA -permute_reg_lambda $PERMUTE_REG_LAMBDA


