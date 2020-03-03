ARCH=transformer_iwslt_de_en
DATA_PATH=data-bin/iwslt14.tokenized.de-en.joined
export PYTHONPATH=$PYTHONPATH:./plot/
LR=0.0005
GPU=1
WU=4000 

for SEED in 1234 
do

echo "seed=$SEED"
echo "warm up updates=$WU"
OUTPUT_PATH=checkpoints/IWSLT/post-norm-with-wu-${LR}-seed$SEED
#OUTPUT_PATH=checkpoints/deen_transformer_fp_822
RESULT_PATH=results/IWSLT/post-norm-with-wu-${LR}-seed$SEED
#RESULT_PATH=results/IWSLT/zgx-deen-${LR}-seed$SEED

mkdir -p $OUTPUT_PATH
mkdir -p $RESULT_PATH
echo "cleaning h5 file"
# rm $OUTPUT_PATH/*.h5

#--encoder-normalize-before --decoder-normalize-before \
CUDA_VISIBLE_DEVICES=$GPU   python3 trans_plot.py   $DATA_PATH  \
    --dir-type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot\
    --x=-1:1:51 --y=-1:1:51 --ngpu 1 --valid-subset valid\
    --seed $SEED --model-file $OUTPUT_PATH/checkpoint_best.pt  \
    -a $ARCH  --share-all-embeddings \
    --optimizer adam --lr $LR \
    -s de -t en \
    --clip-norm 0.0 \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 \
    --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --max-update 50000 \
    --warmup-updates $WU --warmup-init-lr '5e-4' \
    --adam-betas '(0.9, 0.98)' --save-dir $OUTPUT_PATH --restore-file $OUTPUT_PATH/checkpoint_last.pt --reset-optimizer \
    --no-progress-bar --log-interval 100 \
    --ddp-backend=no_c10d 2>&1 | tee -a $OUTPUT_PATH/train_log.txt 
    
done 
