ARCH=transformer_iwslt_de_en
DATA_PATH=data-bin/iwslt14.tokenized.de-en.joined
export PYTHONPATH=$PYTHONPATH:./plot/
LR=0.0005
GPU="0"
WU=4000 
NORM="pre"
WITHWU='nowu'
for SEED in 1234 2345 3456 3456 5678  
do

echo "seed=$SEED"
echo "warm up updates=$WU"

OUTPUT_PATH=checkpoints/IWSLT/$NORM-norm-$WITHWU-${LR}-seed$SEED
#OUTPUT_PATH=checkpoints/deen_transformer_fp_822
RESULT_PATH=results/IWSLT/$NORM-norm-$WITHWU-${LR}-seed$SEED
#RESULT_PATH=results/IWSLT/zgx-deen-${LR}-seed$SEED

mkdir -p $OUTPUT_PATH
mkdir -p $RESULT_PATH
#echo "cleaning h5 file"
# rm $OUTPUT_PATH/*.h5 # ckpt_best_init_valid_normalize_bert_weights_xignore=biasbn_xnorm=filter_inital_direction_normlize_bert_yignore=biasbn_ynorm=filter.h5
#    --dir-file "$OUTPUT_PATH/ckpt_best_init_valid_normalize_bert_weights_xignore=biasbn_xnorm=filter_inital_direction_normlize_bert_yignore=biasbn_ynorm=filter.h5" \
#    # --dir-file 'post-nowu-dir.h5'\
CUDA_VISIBLE_DEVICES=$GPU  python3 plot_loss_surface.py   $DATA_PATH --init-model  --normalize-bert --surf-file "$OUTPUT_PATH/ckpt_init_loss_41x41.h5" \
    --encoder-normalize-before --decoder-normalize-before \
    --dir-type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot\
    --x=-2:2:41 --y=-2:2:41 --ngpu 1 --valid-subset valid\
    --seed $SEED --model-file $OUTPUT_PATH/initpoint_valid_normalize_bert \
    -a $ARCH  --share-all-embeddings \
    --optimizer adam --lr $LR \
    -s de -t en \
    --clip-norm 0.0 \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 \
    --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --max-update 50000 \
    --warmup-updates $WU --warmup-init-lr '5e-4' \
    --adam-betas '(0.9, 0.98)' --save-dir $OUTPUT_PATH --restore-file $OUTPUT_PATH/checkpoint_best_fake.pt --reset-optimizer \
    --no-progress-bar --log-interval 100 \
    --ddp-backend=no_c10d 2>&1 | tee -a $OUTPUT_PATH/train_log.txt 
    
done 
