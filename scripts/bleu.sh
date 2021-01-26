BLEU_SCRIPT=../../attention_explanation/data/fairseq_de_en/mosesdecoder/scripts/generic/multi-bleu.perl

# It's for internal purposes only since we have tokenized the reference by ourself

TRANSLATION_FILE=pred.txt
REFERENCE_FILE=../../attention_explanation/data/fairseq_de_en/iwslt14.tokenized.de-en/valid.en
#REFERENCE_FILE=../cs_data/test.en

#TRANSLATION_FILE=../sig_test/R3.txt

#TRANSLATION_FILE=ready
#REFERENCE_FILE=../datasets/iwslt17/de-en/test.tok.en

echo "Calculating bleu using multi-bleu (for internal purpose only)"

perl $BLEU_SCRIPT $REFERENCE_FILE < $TRANSLATION_FILE
