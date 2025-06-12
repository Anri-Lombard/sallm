#!/bin/sh
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=8 --gres=gpu:ampere:4
#SBATCH --time=48:00:00
#SBATCH --job-name="sallm_transformer_best_final"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

module load python/miniconda3-py3.12

ADAM_BETA1=0.71386
ADAM_BETA2=0.98971
BATCH_SIZE=196608
COOLDOWN_FRAC=0.44139
EMBED_LR=0.79122
HEAD_LR=0.0024099
HIDDEN_LR=0.028399
MODEL_DIM=512
NUM_HEADS=8
NUM_LAYERS=6
SCALAR_LR=0.074012
SEQ_LEN=512
VOCAB_SIZE=50257
ADAM_EPS=1e-10
MOMENTUM_START=0.85
MOMENTUM_END=0.95

EPOCHS=60
TRAIN_FILES="/scratch/lmbanr001/masters/dataset/filtered_data_binary_custom_tokenizer/custom_train_*.bin"
VAL_FILES="/scratch/lmbanr001/masters/dataset/filtered_data_binary_custom_tokenizer/custom_val_*.bin"
VAL_LOSS_EVERY=2000
WANDB_PROJECT="sallm-final-runs"
WANDB_ENTITY="anri-m-lombard"
WANDB_NAME="final_best_125m_e${NUM_LAYERS}h${NUM_HEADS}d${MODEL_DIM}_bash_$(date +%Y%m%d_%H%M%S)"

CHECKPOINT_DIR="/scratch/lmbanr001/masters/sallm/trained_models/final/checkpoints_bash_$(date +%Y%m%d_%H%M%S)"
SAVE_EPOCHS="10,20,30,40,60"
SAVE_BEST_CHECKPOINT_FLAG="--save_best_checkpoint"

mkdir -p "$CHECKPOINT_DIR"

echo "Starting training run..."
torchrun --standalone --nproc_per_node=4 pretrain.py \
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
	--wandb_name "$WANDB_NAME" \
	--checkpoint_dir "$CHECKPOINT_DIR" \
	--save_epochs "$SAVE_EPOCHS" \
	$SAVE_BEST_CHECKPOINT_FLAG

echo "Training run finished."
