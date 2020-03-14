ARCH=transformer_iwslt_de_en
DATA_PATH=data-bin/iwslt14.tokenized.de-en.joined
export PYTHONPATH=$PYTHONPATH:./plot/
LR=0.0005
GPU=2
WU=4000
NORM='post'
WITHWU='nowu'
for SEED in 1234 
do

echo "seed=$SEED"
echo "warm up updates=$WU"
OUTPUT_PATH=checkpoints/IWSLT/$NORM-norm-$WITHWU-${LR}-seed$SEED
#OUTPUT_PATH=checkpoints/deen_transformer_fp_822
RESULT_PATH=results/IWSLT/$NORM-norm-$WITHWU-${LR}-seed$SEED
#RESULT_PATH=results/IWSLT/zgx-deen-${LR}-seed$SEED
BEST_PATH=checkpoints/IWSLT/$NORM-norm-with-wu-$LR-seed1234
DIR_PATH=checkpoints/IWSLT/$NORM-norm-with-wu-$LR-seed1234
mkdir -p $OUTPUT_PATH
mkdir -p $RESULT_PATH
# echo "cleaning h5 file"
# rm $OUTPUT_PATH/*.h5

 #  --dir-file "$DIR_PATH/ckpt_best_init_train_weights_xignore=biasbn_xnorm=filter_inital_direction_yignore=biasbn_ynorm=filter.h5"\
#      #--encoder-normalize-before --decoder-normalize-before \
#\i
CUDA_VISIBLE_DEVICES=$GPU   python3 plot_trajectory.py  $DATA_PATH  --start-epoch 1 --end-epoch 40 --cpu\
    --dir-type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot\
    --x=-4:4:41 --y=-4:4:41 --ngpu 1 --valid-subset valid\
    --model-folder $OUTPUT_PATH \
    --seed $SEED \
    --dir-file "$DIR_PATH/PCA_weights_save_epoch=1/directions.h5" \
    -a $ARCH  --share-all-embeddings \
    --optimizer adam --lr $LR \
    -s de -t en \
    --clip-norm 0.0 \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 \
    --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --max-update 50000 \
    --warmup-updates $WU --warmup-init-lr '5e-4' \
    --adam-betas '(0.9, 0.98)' --save-dir $OUTPUT_PATH --restore-file $BEST_PATH/checkpoint_best_fake.pt --reset-optimizer \
    --no-progress-bar --log-interval 100 \
    --ddp-backend=no_c10d 2>&1 | tee -a $OUTPUT_PATH/plot_trace_log.txt

done
