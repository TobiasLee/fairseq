#!/bin/sh
ARCH=transformer_iwslt_de_en
DATA_PATH=data-bin/iwslt14.tokenized.de-en.joined

SEED=1
WU=8000
[ $# -gt 0 ] && SEED=$1
echo "seed=$SEED"
echo "warm up updates=$WU"
OUTPUT_PATH=checkpoints/IWSLT/bln_wu${WU}_seed$SEED
RESULT_PATH=results/IWSLT/bln_wu${WU}_seed$SEED

mkdir -p $OUTPUT_PATH
mkdir -p $RESULT_PATH
# 500/1k/2k init_lr 1e-7 会爆 --fp16
CUDA_VISIBLE_DEVICES=0 python3 train.py $DATA_PATH  \
    --seed $SEED \
    -a $ARCH  --share-all-embeddings \
    --optimizer adam --lr 0.0005 \
    -s de -t en \
    --clip-norm 0.0 \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 \
    --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --max-update 50000 \
    --warmup-updates $WU --warmup-init-lr '1e-07' \
    --adam-betas '(0.9, 0.98)' --save-dir $OUTPUT_PATH \
    --no-progress-bar --log-interval 100 \
    --ddp-backend=no_c10d --fp16 \
    | tee -a $OUTPUT_PATH/train_log.txt 
    
    

#CUDA_VISIBLE_DEVICES=1 python3 train.py $DATA_PATH \
#  --seed $SEED \
#  -s de -t en \
#  -a $ARCH --share-all-embeddings \
#  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.0001 \
#  --lr 0.001 --min-lr '1e-09' \
#  --clip-norm 0.0 --dropout 0.3 \
#  --lr-scheduler inverse_sqrt --warmup-init-lr '1e-07' --warmup-updates 4000 \
#  --max-tokens 4096 \
#  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#  --max-update 50000 \
#  --save-dir $OUTPUT_PATH \
#  --ddp-backend=no_c10d  --fp16 \
#  --no-progress-bar --log-interval 100 \
#  | tee -a $OUTPUT_PATH/train_log.txt
  
python3 scripts/average_checkpoints.py --inputs $OUTPUT_PATH \
  --num-epoch-checkpoints 10 --output $OUTPUT_PATH/avg_10.pt
   
python3 generate.py $DATA_PATH \
  --path $OUTPUT_PATH/avg_10.pt \
  --log-format simple \
  --batch-size 128 --beam 5 --remove-bpe --lenpen 1.0 \
  > $RESULT_PATH/avg_10.txt
  
python3 generate.py $DATA_PATH \
  --path $OUTPUT_PATH/checkpoint_best.pt \
  --log-format simple \
  --batch-size 128 --beam 5 --remove-bpe --lenpen 1.0 \
  > $RESULT_PATH/checkpoint_best.txt
