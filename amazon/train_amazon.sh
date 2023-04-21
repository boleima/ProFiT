#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-0}
DATA_DIR=${3:-"$REPO/data/"}
OUT_DIR=${4:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='amazon'
NUM_SAMPLE=256
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
else
  BATCH_SIZE=1
  GRAD_ACC=1
fi
  #BATCH_SIZE=8
  #GRAD_ACC=4

SAVE_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}-K${NUM_SAMPLE}/"
mkdir -p $SAVE_DIR

# +
python $PWD/run_classify_amazon.py \
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
  --save_steps `expr 5 \* $NUM_SAMPLE` \
  --eval_all_checkpoints \
  --log_file 'train' \
  --predict_languages $LANGS \
  --save_only_best_checkpoint \
  --overwrite_output_dir \
  --overwrite_cache \
  --num_sample $NUM_SAMPLE \


#--model_name_or_path "$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}/checkpoint-best" \
  #--init_checkpoint "$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}/checkpoint-best" 
  

#
#
#--do_train \
#--do_eval \   
#--eval_test_set 
