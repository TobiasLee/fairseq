DATA_PATH=data-bin/fairseq-iwslt.tokenized.en-vi.bpe10k 
ARCH=transformer_wmt_en_de
GPU="2"
for seed in 1234 2345 3456 4567 5678 
do
OUTPUT_PATH=checkpoints/en-vi/adawu_seed$seed
RESULT_PATH=results/en-vi/adawu_seed$seed

mkdir -p $OUTPUT_PATH
mkdir -p $RESULT_PATH

CUDA_VISIBLE_DEVICES=$GPU python3 train.py $DATA_PATH  -a $ARCH --share-decoder-input-output-embed \
    --seed $seed  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0  \
    --lr 1e-3 --lr-scheduler adaptive_warmup \
    --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --save-dir $OUTPUT_PATH \
    --tensorboard-logdir $OUTPUT_PATH --max-epoch 20 --max-update 0 \
    --log-format simple --log-interval 500   2>&1 | tee $OUTPUT_PATH/train_log.txt


CUDA_VISIBLE_DEVICES=$GPU python3 scripts/average_checkpoints.py --inputs $OUTPUT_PATH \
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
