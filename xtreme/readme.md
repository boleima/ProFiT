*Still in construction...*
## Usage
### 1. Download dataset
Use the bash file `download_data.sh` to download the required data sets:
```
bash scripts/download_data.sh
```
### 2. Run baseline
Use the bash file `train_{task_name}.sh` to run the baseline.
For example, run the baseline for the task xnli:
```
bash scripts/train_xnli.sh
```

### 3. Run prompt-based training
Use the bash file `train_prompt_{task_name}.sh` to run the prompt-based training.
For example, run the prompt-based training for the task xnli:
```
bash scripts/train_prompt_xnli.sh
```
