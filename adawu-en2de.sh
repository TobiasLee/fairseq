ARCH=transformer_iwslt_de_en
DATA_PATH=data-bin/iwslt14.tokenized.de-en.joined

LR=0.0005
GPU=1
WU=4000

for SEED in 1234 2345 3456 4567 5678 
do
echo "seed=$SEED"
echo "warm up updates=$WU"
OUTPUT_PATH=checkpoints/en-de/adawu_seed$seed
RESULT_PATH=results/en-de/adawu_seed$seed
mkdir -p $OUTPUT_PATH
mkdir -p $RESULT_PATH


CUDA_VISIBLE_DEVICES=$GPU python3 train.py $DATA_PATH  \
    --seed $SEED   \
    -a $ARCH  --share-all-embeddings  \
    --optimizer adam --lr $LR  \
    -s de -t en \
    --clip-norm 0.0 \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 \
    --lr-scheduler adaptive_warmup --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --max-update 60000 \
    --warmup-updates $WU --warmup-init-lr $LR  \
    --adam-betas '(0.9, 0.98)' --save-dir $OUTPUT_PATH \
    --no-progress-bar --log-interval 100 \
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

