ARCH=transformer_iwslt_de_en
DATA_PATH=data-bin/iwslt14.tokenized.de-en.joined
export PYTHONPATH=$PYTHONPATH:./plot/
LR=0.0005
GPU=1
WU=4000
NORM="post"
WARMUP="with-wu"

# what we need
# a surface file, to show the trajectory of optimization
# around best ckpt --encoder-normalize-before --decoder-normalize-before 
#\
for SEED in 1234
do

echo "seed=$SEED, norm type=$NORM, warmup type=$WARMUP"
OUTPUT_PATH=checkpoints/IWSLT/$NORM-norm-$WARMUP-${LR}-seed$SEED
RESULT_PATH=results/IWSLT/$NORM-norm-$WARMUP-${LR}-seed$SEED

mkdir -p $OUTPUT_PATH
mkdir -p $RESULT_PATH

CUDA_VISIBLE_DEVICES=$GPU   python3 plot_contour_trace.py  $DATA_PATH  --seed $SEED \
    --dir-type weights --restore-file $OUTPUT_PATH/checkpoint_best.pt --dir-folder $OUTPUT_PATH \
    --x=-2:2:81 --y=-2:2:81 --ngpu 1 --valid-subset valid\
    --model-folder $OUTPUT_PATH  \
    -a $ARCH  --share-all-embeddings \
    --optimizer adam --lr $LR \
    -s de -t en \
    --clip-norm 0.0 \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 \
    --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --max-update 50000 \
    --warmup-updates $WU --warmup-init-lr '5e-4' \
    --adam-betas '(0.9, 0.98)' --save-dir $OUTPUT_PATH  --reset-optimizer \
    --no-progress-bar --log-interval 100 \
    --ddp-backend=no_c10d 2>&1 | tee -a $OUTPUT_PATH/plot_surface_trace_log.txt

# proj init model on the surface file already plotted
NOWU_PATH=checkpoints/IWSLT/$NORM-norm-nowu-${LR}-seed$SEED
CUDA_VISIBLE_DEVICES=$GPU   python3 plot_contour_trace.py  $DATA_PATH  --seed $SEED \
    --dir-type weights --restore-file $OUTPUT_PATH/checkpoint_best.pt --init-model --dir-folder $OUTPUT_PATH\
    --x=-2:2:81 --y=-2:2:81 --ngpu 1 --valid-subset valid\
    --model-folder $NOWU_PATH  \
    -a $ARCH  --share-all-embeddings \
    --optimizer adam --lr $LR \
    -s de -t en \
    --clip-norm 0.0 \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 \
    --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --max-update 50000 \
    --warmup-updates $WU --warmup-init-lr '5e-4' \
    --adam-betas '(0.9, 0.98)' --save-dir $OUTPUT_PATH  --reset-optimizer \
    --no-progress-bar --log-interval 100 \
    --ddp-backend=no_c10d 2>&1 | tee -a $OUTPUT_PATH/plot_surface_trace_log.txt
done
