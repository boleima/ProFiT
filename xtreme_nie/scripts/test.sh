  
#   PATTERN_IDS=${1:-0 1 2 3 4}
#   for PATTERN_ID in $PATTERN_IDS
#   do 
#     python run_baseline/run_prompt_classify.py \
#       --model_type bert\
#       --model_name_or_path /mounts/work/nie/projects/xtreme/outputs/xnli/$PATTERN_ID/bert-base-multilingual-cased-LR2e-5-epoch5-MaxLen128-PatternID$PATTERN_ID-{retrieval}/checkpoint-best \
#       --task_name xnli \
#       --do_predict \
#       --data_dir download/xnli \
#       --gradient_accumulation_steps 4 \
#       --per_gpu_train_batch_size 8 \
#       --learning_rate 2e-5 \
#       --num_train_epochs 5 \
#       --max_seq_length 128 \
#       --output_dir outputs/xnli/test_results_$PATTERN_ID \
#       --save_steps 1000 \
#       --log_file 'train' \
#       --predict_languages ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh \
#       --save_only_best_checkpoint \
#       --overwrite_output_dir \
#       --overwrite_cache \
#       --eval_test_set \
#       --pattern_id $PATTERN_ID
# done

test(){
  echo ${1}
  echo ${TEST}
}

TESTS=(1 2)
for TEST in "${TESTS[@]}"
  do test $TEST
done