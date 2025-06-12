#!/bin/sh
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=4 --gres=gpu:l40s:4
#SBATCH --time=48:00:00
#SBATCH --job-name="sallm_transformer"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

module load python/miniconda3-py3.12

# Set hyperparameters
TRAIN_FILES="/scratch/lmbanr001/masters/dataset/filtered_data_binary_custom_tokenizer/custom_train_*.bin"
VAL_FILES="/scratch/lmbanr001/masters/dataset/filtered_data_binary_custom_tokenizer/custom_val_*.bin"
BATCH_SIZE=65536 # 64 * 1024
SEQ_LEN=2048
EPOCHS=30

# Model architecture parameters
VOCAB_SIZE=50257
NUM_LAYERS=8
NUM_HEADS=8
MODEL_DIM=512

# Optimizer parameters
HEAD_LR=0.008
EMBED_LR=0.6
SCALAR_LR=0.04
HIDDEN_LR=0.025
MOMENTUM_START=0.85
MOMENTUM_END=0.95
ADAM_BETA1=0.8
ADAM_BETA2=0.95
ADAM_EPS=1e-10
COOLDOWN_FRAC=0.4
VAL_LOSS_EVERY=2000

WANDB_PROJECT="sallm"
WANDB_ENTITY="anri-m-lombard"
WANDB_NAME="transformer-experiment-1"

torchrun --standalone --nproc_per_node=4 train_transformers.py \
	--train_files "$TRAIN_FILES" \
	--val_files "$VAL_FILES" \
	--batch_size $BATCH_SIZE \
	--seq_len $SEQ_LEN \
	--epochs $EPOCHS \
	--vocab_size $VOCAB_SIZE \
	--num_layers $NUM_LAYERS \
	--num_heads $NUM_HEADS \
	--model_dim $MODEL_DIM \
	--head_lr $HEAD_LR \
	--embed_lr $EMBED_LR \
	--scalar_lr $SCALAR_LR \
	--hidden_lr $HIDDEN_LR \
	--momentum_start $MOMENTUM_START \
	--momentum_end $MOMENTUM_END \
	--adam_beta1 $ADAM_BETA1 \
	--adam_beta2 $ADAM_BETA2 \
	--adam_eps $ADAM_EPS \
	--cooldown_frac $COOLDOWN_FRAC \
	--val_loss_every $VAL_LOSS_EVERY \
	--wandb_project "$WANDB_PROJECT" \
	--wandb_entity "$WANDB_ENTITY" \
	--wandb_name "$WANDB_NAME"
