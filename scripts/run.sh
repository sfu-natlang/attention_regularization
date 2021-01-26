#!/bin/bash

#MODEL_PATH=/local-scratch/pooya/attention_regularization/models/default_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/experiments/baseline/1/best.pt

#MODEL_PATH=/local-scratch/pooya/attention_regularization/ft_from_b3/model_step_5000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/uni_0.285-perm_0.142-zom_0.568-R1-C2.5/model_step_5000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/cs-en/baseline/1/model_step_20000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/cs-en/cs_all_R1-C2.5/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/cs-en/cs-ent-0.04/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/uni-R1-C2.5/model_step_5000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/winter/de-ft_from_b1_reward/model_step_1000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/de/baseline_winter/2/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/de/restart_baseline_winter/3/model_step_2000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/winter/de-ft_from_b3_reward/model_step_1000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/de/restart_baseline_winter/1/model_step_3000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/winter/de-ft_from_b1_reward/model_step_2000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/winter/de-ft_from_r1_reward/model_step_4000.pt
#TEST_SRC=../../attention_explanation/data/fairseq_de_en/iwslt14.tokenized.de-en/valid.de
#MODEL_PATH=/local-scratch/pooya/attention_regularization/de/restart_baseline_winter/2/model_step_1000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/aacl/de/2/ft_from_baseline/model_step_9000.pt
MODEL_PATH=/local-scratch/pooya/attention_regularization/aacl/de/1/ft_from_baseline/model_step_3000.pt
TEST_SRC=../../attention_explanation/data/fairseq_de_en/iwslt14.tokenized.de-en/test.de

#TEST_TRG=../../attention_explanation/data/fairseq_de_en/iwslt14.tokenized.de-en/valid.en
#TEST_SRC=../cs_data/test.de
#TEST_TRG=../cs_data/test.en

#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/experiments/iwslt.de-en/baseline/1/model_step_10000.pt
#lign_debugTEST_SRC=../datasets/iwslt17/de-en/test.tok.tc.de 

python ../OpenNMT-py/translate.py -model $MODEL_PATH -src $TEST_SRC -gpu 1 -batch_size 64 -beam_size 1 #-beam_size 5 #-beam_size 1



