#!/bin/bash

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-1,2,3,4,5,6,7,0}
DATA_DIR=${3:-"$REPO/data/"}
OUT_DIR=${4:-"$REPO/outputs/"}
PATTERN_ID=${5:-0}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='amazon'
LR=2e-5
EPOCH=5
MAXL=128
LANGS="de","en","es","fr","ja","zh"
LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=2
  GRAD_ACC=16
  LR=3e-5
else
  BATCH_SIZE=8
  GRAD_ACC=4
  LR=2e-5
fi

SAVE_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}-PatternID${PATTERN_ID}/"
mkdir -p $SAVE_DIR

python $PWD/run_prompt_classify_amazon.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --train_language en \
  --task_name $TASK \
  --do_train \
  --do_eval \
  --do_predict \
  --data_dir $DATA_DIR/${TASK} \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --max_seq_length $MAXL \
  --output_dir $SAVE_DIR/ \
  --save_steps 1000 \
  --eval_all_checkpoints \
  --log_file 'train' \
  --predict_languages $LANGS \
  --save_only_best_checkpoint \
  --overwrite_output_dir \
  --overwrite_cache \
  --pattern_id $PATTERN_ID \
  --num_priming 1 \