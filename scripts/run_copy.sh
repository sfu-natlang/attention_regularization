#MODEL_PATH=/local-scratch/pooya/attention_regularization/models/default_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/experiments/baseline/1/best.pt

#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/baseline/3/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg1/lambda0.01/1/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg1/lambda0.01/2/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg_all/uni_0.0010-perm_0.0010-zo_max_0.0010/1/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg_all/uni_0.00105-perm_0.00105-zo_max_0.00105/1/model_step_15000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/reg_all/uni_0.001-perm_0.001-zo_max_0.001/2/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/iwslt-de-en/experiments/baseline/5/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/cs-en/baseline/2/model_step_10000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/aacl/cs-en/restart/2/model_step_7000.pt
#MODEL_PATH=/local-scratch/pooya/attention_regularization/aacl/cs-en/ft2/model_step_7000.pt

#TEST_SRC=../../attention_explanation/data/fairseq_de_en/iwslt14.tokenized.de-en/test.de
#TEST_TRG=../../attention_explanation/data/fairseq_de_en/iwslt14.tokenized.de-en/test.en
TEST_SRC=../cs_data/test.cs
TEST_TRG=../cs_data/test.en

#MODEL_PATH=/local-scratch/pooya/attention_regularization/new/experiments/iwslt.de-en/baseline/1/model_step_10000.pt
#TEST_SRC=../datasets/iwslt17/de-en/test.tok.tc.de 

CUDA_VISIBLE_DEVICES=1 python ../OpenNMT-py/translate.py -model $MODEL_PATH -src $TEST_SRC -tgt $TEST_TRG -gpu 0 -batch_size 32 -beam_size 1 #-beam_size 5 #-beam_size 1


