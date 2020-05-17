ARCH=transformer_iwslt_de_en
DATA_PATH=data-bin/iwslt14.tokenized.de-en.joined

# adawu param
BOUND_LO=0.5
BOUND_HI=2.0
BETA3=0.999
BETA4=0.995

LR=0.0005
GPU=0
WU=4000

for SEED in 1234 2345 3456 4567 5678
do
echo "seed=$SEED"
echo "warm up updates=$WU"
OUTPUT_PATH=checkpoints/de-en/adawu_term_seed${SEED}_WU${WU}_B3${BETA3}_B4${BETA4}
RESULT_PATH=results/de-en/adawu_term_seed${SEED}_WU${WU}_B3${BETA3}_B4${BETA4}
mkdir -p $OUTPUT_PATH
mkdir -p $RESULT_PATH


CUDA_VISIBLE_DEVICES=$GPU python3 train.py $DATA_PATH \
    --seed $SEED  \
    -a $ARCH  --share-all-embeddings  \
    --optimizer adam_adawu --lr $LR  -s de -t en \
    --tensorboard-logdir $OUTPUT_PATH --empty-cache-freq 500\
    --clip-norm 0.0 --keep-last-epochs 30 \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 \
    --lr-scheduler adaptive_warmup_term --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --bound-lo $BOUND_LO --bound-hi $BOUND_HI \
    --beta3 $BETA3 --beta4 $BETA4 \
    --max-update 0  --max-update 60000 \
    --warmup-updates $WU --warmup-init-lr $LR  \
    --adam-betas '(0.9, 0.98)' --save-dir $OUTPUT_PATH \
    --no-progress-bar --log-interval 50 \
    --ddp-backend=no_c10d 2>&1 | tee -a $OUTPUT_PATH/train_log.txt


python3 scripts/average_checkpoints.py --inputs $OUTPUT_PATH \
  --num-epoch-checkpoints 10 --output $OUTPUT_PATH/avg_10.pt

CUDA_VISIBLE_DEVICES=$GPU python3 generate.py $DATA_PATH \
  --path $OUTPUT_PATH/avg_10.pt \
  --log-format simple \
  --batch-size 128 --beam 5 --remove-bpe --lenpen 1.0 \
  > $RESULT_PATH/avg_10.txt

CUDA_VISIBLE_DEVICES=$GPU python3 generate.py $DATA_PATH \
  --path $OUTPUT_PATH/checkpoint_best.pt \
  --log-format simple \
  --batch-size 128 --beam 5 --remove-bpe --lenpen 1.0 \
  > $RESULT_PATH/checkpoint_best.txt

done

