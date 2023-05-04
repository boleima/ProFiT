#!/bin/bash

REPO=$PWD
GPU=${1:-0,1,2,3,4,5,6,7}
MODEL=${2:-bert-base-multilingual-cased}
DATA_DIR=${3:-"amazon_reviews_multi"}
OUT_DIR=${4:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='amazon'
EPOCH=5
MAXL=128
SEEDS=(10 42 421 520 1218)

LANGS="de,en,es,fr,ja,zh"
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
  LR=1e-5
fi


runfewshot(){
  # SAVE_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}/"
  NAME="test_results_baseline-epoch${EPOCH}-seed${1}"  
  SAVE_DIR="${OUT_DIR}/${TASK}/${NAME}/"
  RESULT_FILE="results_${TASK}_bl_full.csv"
  mkdir -p $SAVE_DIR
  python $PWD/run_baseline/run_classify.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --train_language en \
    --task_name $TASK \
    --do_train \
    --do_predict \
    --data_dir $DATA_DIR \
    --gradient_accumulation_steps $GRAD_ACC \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --num_train_epochs $EPOCH \
    --max_seq_length $MAXL \
    --output_dir $SAVE_DIR/ \
    --log_file 'train' \
    --predict_languages $LANGS \
    --save_only_best_checkpoint \
    --overwrite_output_dir \
    --overwrite_cache \
    --eval_test_set $LC \
    --seed ${1} \
    --early_stopping
  python $PWD/results_to_csv.py \
    --input_path "${SAVE_DIR}test_results.txt" \
    --save_path $RESULT_FILE \
    --name $NAME
}

for SEED in "${SEEDS[@]}"
do
  runfewshot $SEED
done