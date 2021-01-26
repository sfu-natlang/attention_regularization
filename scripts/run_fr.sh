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
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/zom_unk_pretrained_1-run_1/model_step_6000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg_all/uni_0.33-perm_0.33-zo_max_0.33/1/model_step_1000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg_all/uni_0.33-perm_0.33-zo_max_0.33/1/model_step_8000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg_all/uni_0.33-perm_0.33-zo_max_0.33/1/model_step_13000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/zom_1-R1-C2.5/model_step_15000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/uni-R1-C2.5/model_step_5000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/ent-reg_run_1-lambda_0.04/model_step_10000.pt
#MODEL_PATH=/cs/natlang-expts/pooya/attention_regularization/pretrained/pretrained.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/fr-en/baseline/1/model_step_25000.pt
MODEL_PATH=/local-scratch/pooya/attention_regularization/cs-en/baseline/2/model_step_5000.pt
TEST_SRC=../fr_data/valid.cs
TEST_TRG=../fr_data/valid.en

#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/experiments/iwslt.de-en/baseline/1/model_step_10000.pt
#lign_debugTEST_SRC=../datasets/iwslt17/de-en/test.tok.tc.de 

python ../OpenNMT-py/translate.py -model $MODEL_PATH -src $TEST_SRC -tgt $TEST_TRG -gpu 0 -batch_size 64 -beam_size 1 #-beam_size 5 #-beam_size 1



