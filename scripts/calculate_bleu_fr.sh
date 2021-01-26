BLEU_SCRIPT=../../attention_explanation/data/fairseq_de_en/mosesdecoder/scripts/generic/multi-bleu.perl

# It's for internal purposes only since we have tokenized the reference by ourself

TRANSLATION_FILE=pred.txt
REFERENCE_FILE=../fr_data/valid.en

#TRANSLATION_FILE=ready
#REFERENCE_FILE=../datasets/iwslt17/de-en/test.tok.en

echo "Calculating bleu using multi-bleu (for internal purpose only)"

perl $BLEU_SCRIPT $REFERENCE_FILE < $TRANSLATION_FILE
