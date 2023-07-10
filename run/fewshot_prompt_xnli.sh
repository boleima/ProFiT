#!/bin/bash

REPO=$PWD
GPU=${1:-1,2,3,4,5,6,7,0}
MODEL=${2:-bert-base-multilingual-cased}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}
PATTERN_ID=${5:-1}
MODEL_TYPE=${6:-bert}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='xnli'
EPOCH=50
MAXL=128
LANGS="ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh"
LC=""
NUM_SAMPLES=(1 2 4 8 16 32 64 128 256 512 1024)


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
  BATCH_SIZE=1
  GRAD_ACC=2
  LR=1e-5
fi

SEED=10


runfewshot(){
  NAME="${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}-Pattern${PATTERN_ID}-${NUM_SAMPLE}shot"
  SAVE_DIR="$OUT_DIR/$TASK/$PATTERN_ID/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}-Pattern${PATTERN_ID}-${NUM_SAMPLE}shot/"
  RESULT_FILE="results_${TASK}_prompt.csv"
  mkdir -p $SAVE_DIR
  python $PWD/py_scripts/run_prompt_classify.py \
    --seed ${SEED} \
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
    --log_file 'train' \
    --predict_languages $LANGS \
    --save_only_best_checkpoint \
    --overwrite_output_dir \
    --overwrite_cache \
    --eval_test_set $LC \
    --pattern_id $PATTERN_ID \
    --num_sample ${1} \
    --early_stopping
    --init_checkpoint outputs/xnli/1/xlm-roberta-base-LR1e-5-epoch50-MaxLen128-Pattern1-1024shot/checkpoint-best/
	
  python $PWD/results_to_csv.py \
    --input_path "${SAVE_DIR}test_results.txt" \
    --save_path $RESULT_FILE \
    --name $NAME
}



for NUM_SAMPLE in "${NUM_SAMPLES[@]}"
do
  runfewshot $NUM_SAMPLE
done
