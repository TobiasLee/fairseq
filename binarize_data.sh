# Binarize the dataset
TEXT=examples/translation/iwslt14.tokenized.de-en # examples/iwslt14.tokenized.de-en/
fairseq-preprocess \
    --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --joined-dictionary \
    --destdir data-bin/iwslt14.tokenized.de-en-joined --thresholdtgt 0 --thresholdsrc 0 \
    --workers 60
