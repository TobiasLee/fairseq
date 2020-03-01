ARCH=transformer_iwslt_de_en
DATA_PATH=data-bin/iwslt14.tokenized.de-en-joined

LR=0.0005
GPU=1
WU=4000 
# SEED=8888 

for SEED in 1 2 3 4 5 1234 4321 3456 6543 8888 
do

for NUM_LAYER in 6 8 10 12 14 
do 


echo "seed=$SEED" 
echo "layer num=$NUM_LAYER"
echo "warm up updates=$WU"
OUTPUT_PATH=checkpoints/IWSLT/pre-norm-test-seed-$SEED-layer-$NUM_LAYER
RESULT_PATH=results/IWSLT/pre-norm-test-seed$SEED-layer-$NUM_LAYER

mkdir -p $OUTPUT_PATH
mkdir -p $RESULT_PATH

CUDA_VISIBLE_DEVICES=$GPU python3 train.py $DATA_PATH  \
    --seed $SEED  \
    -a $ARCH  --share-all-embeddings  --encoder-normalize-before --decoder-normalize-before \
    --optimizer adam --lr $LR \
    -s de -t en  --encoder-layers $NUM_LAYER --decoder-layers $NUM_LAYER \
    --clip-norm 0.0 \
    --label-smoothing 0.1 --dropout 0.1 --max-tokens 4096 \
    --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --max-update 50000 \
    --warmup-updates $WU --warmup-init-lr $LR  \
    --adam-betas '(0.9, 0.98)' --save-dir $OUTPUT_PATH \
    --no-progress-bar --log-interval 100 \
    --ddp-backend=no_c10d 2>&1 | tee -a $OUTPUT_PATH/train_log.txt 


done
done

for SEED in 1 2 3 4 5 1234 4321 3456 6543 8888 
do

for NUM_LAYER in 6 8 10 12 14 
do 


echo "seed=$SEED" 
echo "layer num=$NUM_LAYER"
echo "warm up updates=$WU"
OUTPUT_PATH=checkpoints/IWSLT/post-norm-test-seed-$SEED-layer-$NUM_LAYER
RESULT_PATH=results/IWSLT/post-norm-test-seed-$SEED-layer-$NUM_LAYER

mkdir -p $OUTPUT_PATH
mkdir -p $RESULT_PATH

CUDA_VISIBLE_DEVICES=$GPU python3 train.py $DATA_PATH  \
    --seed $SEED  \
    -a $ARCH  --share-all-embeddings  \
    --optimizer adam --lr $LR \
    -s de -t en --encoder-layers $NUM_LAYER --decoder-layers $NUM_LAYER \
    --clip-norm 0.0 \
    --label-smoothing 0.1 --dropout 0.1 --max-tokens 4096 \
    --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --max-update 50000 \
    --warmup-updates $WU --warmup-init-lr $LR  \
    --adam-betas '(0.9, 0.98)' --save-dir $OUTPUT_PATH \
    --no-progress-bar --log-interval 100 \
    --ddp-backend=no_c10d 2>&1 | tee -a $OUTPUT_PATH/train_log.txt 


done
done 
