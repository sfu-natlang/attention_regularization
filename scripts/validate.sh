#!/bin/bash

#MODEL_PATH=/local-scratch/pooya/attention_regularization/models/default_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/experiments/baseline/1/best.pt

#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/baseline/3/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg1/lambda0.01/1/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg1/lambda0.01/2/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg_all/uni_0.0010-perm_0.0010-zo_max_0.0010/1/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg_all/uni_0.0002-perm_0.0002-zo_max_0.0002/1/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg_all/uni_0.001-perm_0.001-zo_max_0.001/3/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg1-1gpu_run_1-lambda_0.01/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/ent-reg-1gpu_run_1-lambda_0.04/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/ent-reg-1gpu_run_1-lambda_0.02/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg1-1gpu_run_3-lambda_0.01/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/zom_unk_0.00875-run_1/model_step_15000.pt 
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/baseline/1/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/baseline/4/model_step_20000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/zom_unk_pretrained_1-run_1/model_step_4000.pt
MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/zom_unk_pretrained_1-run_1/model_step_6000.pt
DATA_PREFIX=../opennmt_preprocessed_data/iwslt-de-en

TEST_SRC=../../attention_explanation/data/fairseq_de_en/iwslt14.tokenized.de-en/valid.de
TEST_TRG=../../attention_explanation/data/fairseq_de_en/iwslt14.tokenized.de-en/valid.en

#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/experiments/iwslt.de-en/baseline/1/model_step_10000.pt
#lign_debugTEST_SRC=../datasets/iwslt17/de-en/test.tok.tc.de 

python ../OpenNMT-py/validate_forced.py -train_from $MODEL_PATH -data $DATA_PREFIX -world_size 1 -gpu_ranks 0 #-attn_debug #-beam_size 5 #-beam_size 1



