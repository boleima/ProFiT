#!/bin/bash

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-1,2,3,4,5,6,7,0}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}
PATTERN_ID=${5:-1}
NUM_SAMPLE=${6:-8}
MODEL_TYPE=${7:-bert}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='xnli'
LR=2e-5
EPOCH=5
MAXL=128
LANGS="ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh"
LC=""

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=2
  GRAD_ACC=16
  LR=3e-5
else
  BATCH_SIZE=1
  GRAD_ACC=1
  LR=2e-5
fi


SAVE_DIR="$OUT_DIR/$TASK/$PATTERN_ID/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}-Pattern${PATTERN_ID}-${NUM_SAMPLE}shot/"
mkdir -p $SAVE_DIR
python $PWD/run_baseline/run_prompt_classify.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --task_name $TASK \
  --do_train \
  --do_predict \
  --data_dir $DATA_DIR/${TASK} \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --max_seq_length $MAXL \
  --output_dir $SAVE_DIR/ \
  --save_steps 1000 \
  --log_file 'train' \
  --predict_languages $LANGS \
  --save_only_best_checkpoint \
  --overwrite_output_dir \
  --overwrite_cache \
  --eval_test_set $LC \
  --pattern_id $PATTERN_ID \
  --num_sample $NUM_SAMPLE
#  --init_checkpoint outputs/xnli/bert-base-multilingual-cased-LR2e-5-epoch5-MaxLen128-PatternID1/checkpoint-best/