#!/bin/bash

#MODEL_PATH=/local-scratch/pooya/attention_regularization/cs-en/baseline/3/model_step_20000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/cs-en/baseline/1/model_step_15000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/cs-en/restart/1/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/cs-en/restart/3/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/cs-en/cs_ft2/model_step_5000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/cs-en/cs_all_R1-C2.5/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/baseline/1/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/cs-en/baseline/3/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/cs-en/cs_ft3/model_step_5000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/cs-en/cs_ft2/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/cs-en/baseline/3-1k/model_step_6000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/cs-en/ft1-1k/model_step_5000.pt
MODEL_PATH=/local-scratch/pooya/attention_regularization/cs-en/baseline/1-1k/model_step_6000.pt
TEST_SRC=../cs_data/test.cs

#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/experiments/iwslt.de-en/baseline/1/model_step_10000.pt
#lign_debugTEST_SRC=../datasets/iwslt17/de-en/test.tok.tc.de 

python ../OpenNMT-py/translate.py -model $MODEL_PATH -src $TEST_SRC -gpu 0 -batch_size 64 -beam_size 1 #-beam_size 5 #-beam_size 1



