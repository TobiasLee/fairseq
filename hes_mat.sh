ARCH=transformer_iwslt_de_en
DATA_PATH=data-bin/iwslt14.tokenized.de-en.joined
export PYTHONPATH=$PYTHONPATH:./plot/
LR=0.0005
GPU=1
WU=4000

for SEED in 2345 
do

echo "seed=$SEED"
echo "warm up updates=$WU"
OUTPUT_PATH=checkpoints/IWSLT/post-norm-with-wu-$WU-$LR-seed$SEED-hessain
RESULT_PATH=results/IWSLT/post-norm-with-wu-$WU-$LR-seed$SEED-hessain

mkdir -p $OUTPUT_PATH
mkdir -p $RESULT_PATH

#--encoder-normalize-before --decoder-normalize-before \
CUDA_VISIBLE_DEVICES=$GPU python3 hessian_density.py  $DATA_PATH --hessian --cpu\
    --seed $SEED   \
    -a $ARCH  --share-all-embeddings --train-subset valid --valid-subset valid\
    --optimizer adam --lr $LR \
    -s de -t en \
    --clip-norm 0.0 \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 \
    --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --max-update 50000 \
    --warmup-updates $WU --warmup-init-lr '1e-07' \
    --adam-betas '(0.9, 0.98)' --save-dir $OUTPUT_PATH \
    --no-progress-bar --log-interval 100 \
    --ddp-backend=no_c10d 2>&1 | tee -a $OUTPUT_PATH/train_log.txt 
    
    

done 
