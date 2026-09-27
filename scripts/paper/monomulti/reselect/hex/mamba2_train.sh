#!/bin/bash
# Retrain a Mamba2 task adapter with its original recipe, changing only checkpoint handling:
# save every epoch (adapter only), no early stopping, no keep-best. Evidence and the per-label
# recipe notes are in $root/tmp/mm_recipe/RECIPES.md.
set -euo pipefail
umask 022
label="${1:?label}"
source /scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925/jobs/train_common.sh

base=anrilombard/sallm-mamba-125m
base_file="$hf_mamba_blob/pytorch_model.bin"
base_sha="$mamba_sha"
tok="$sallm_tok"
tok_sha="$sallm_tok_sha"
venv="$live_venv"
launcher=accelerate
env_extra=()
deviations="none known"
common=(
  finetune.training.report_to=none
  finetune.hub.enabled=false
  finetune.hub.push_adapter=false
  finetune.hub.push_merged=false
)

case "$label" in
  mamba2_intent_multi)
    code="$code_jul"
    config=mamba_injongointent_all
    original="HEX /scratch/lmbanr001/masters/sallm/checkpoints/intent_recovery_20260729/mamba_clean_r2 (Slurm 1130570, 2026-07-29, A100; kept checkpoint-888 = epoch 8 of 10 by eval_classification/all_f1)"
    recipe_note="config defaults: r16 a32 dropout0 in_proj,x_proj; trainable_token_indices=3 added special tokens; lr 8e-5 cosine warmup0.03 wd0.01 beta2 0.95; bs32 ga2; 10 epochs; bf16; grad ckpt; max_len 2048; lm_eval_p1-5 CYCLE train, ALL val; clean InjongoIntent split (test-text decontaminated, validation = 10% of clean train): train 7045, val 831x5=4155; seed 42, data_seed unset"
    args=("${common[@]}" finetune.wandb.project=sallm-ft "${keep_all[@]}")
    env_extra=("SALLM_INJONGO_OFFLINE_SNAPSHOT=$injongo_snapshot")
    changes="save_total_limit 1->null; load_best_model_at_end true->false; early_stopping_patience 3->null; save_only_model false->true"
    deviations="code is a frozen copy of the live HEX repo taken 2026-09-25: configs, templates, data loaders and metrics are byte-identical to the July run (unchanged since before 2026-07-29 09:04); 16 source files changed later (FILES_CHANGED_AFTER_20260729T0904.txt), all GDN/xLSTM/eval/packing guards that do not touch the Mamba intent path. InjongoIntent jsonl is read from the local HF cache snapshot fe4be388 instead of resolve/main (the dataset has not changed since 2025-01)."
    ;;
  mamba2_intent_mono_eng|mamba2_intent_mono_sot|mamba2_intent_mono_xho|mamba2_intent_mono_zul)
    lang="${label##*_}"
    code="$code_feb"
    config="mamba_injongointent_$lang"
    declare -A ga_orig=([eng]=4 [sot]=8 [xho]=2 [zul]=2)
    ga=$(( ga_orig[$lang] * 2 ))
    original="hub anrilombard/sallm-mamba2-masakhane-injongointent-$lang (pushed 2026-02-17..20; copy /scratch/lmbanr001/masters/sallm_snapshots/full-matrix-retained-bindings-20260916-v1/adapters/mamba2/intent/mono_$lang); no trainer state or log survives, selected epoch unknown"
    recipe_note="per-cell config at git cf0ed06 (identical to d90c0a1, 2026-02-16, the first commit whose LoRA settings match the hub adapter_config incl. zul dropout 0.05); keep-best on eval_classification/all_accuracy, patience 3 in the original; upstream InjongoIntent train + upstream dev (eng has no dev split, so the Feb loader fell back to TEST as in-training eval set)"
    args=("${common[@]}" "finetune.training.gradient_accumulation_steps=$ga" "${keep_all[@]}")
    changes="save_total_limit 1->null; load_best_model_at_end true->false; early_stopping_patience 3->null; save_only_model false->true; report_to wandb->none; hub push off"
    deviations="1 GPU with gradient_accumulation_steps ${ga_orig[$lang]}->$ga: the Feb launcher (scripts/launch_finetune.sh at cf0ed06, #SBATCH --gres=gpu:l40s:2, NUM_PROCS from SLURM_GPUS_ON_NODE) ran 2-process DDP, so this keeps the effective batch; the 2-GPU launch is inferred from the launcher, no log survives. Code state cf0ed06 vs the exact Feb working tree is unverifiable."
    if [[ "$lang" == eng ]]; then
      env_extra=(
        "SALLM_INJONGO_OFFLINE_SNAPSHOT=$injongo_snapshot"
        "SALLM_CLEAN_INJONGO_SPLIT_MODULE=$code_jul/src/main/sallm/data/loaders/injongointent_split.py"
      )
      deviations="$deviations DATA DEVIATION (eng only): the Feb loader had no eng dev split and fell back to TEST for validation, and Feb eng train overlaps test texts. This run keeps the Feb code and every Feb hyperparameter but replaces the InjongoIntent loader with the July clean pipeline (jobs/code/sallm-live-20260925 injongointent_split.py on the HF cache snapshot fe4be388): upstream eng train.jsonl minus rows whose normalized text appears in test.jsonl, then a stable md5 carve of 10% per intent as validation. Train 1046 rows, validation 111 rows (was: train 1779 upstream rows, validation = test 622 rows)."
    fi
    ;;
  mamba2_news_mono_eng)
    code="$code_feb"
    config=mamba_news_eng
    original="hub anrilombard/sallm-mamba2-masakhane-masakhanews-eng@3ea3e4f (pushed 2026-02-16; copy /scratch/lmbanr001/masters/sallm_snapshots/full-matrix-retained-bindings-20260916-v1/adapters/mamba2/news_mono_eng); selected epoch unknown"
    recipe_note="per-cell config at git cf0ed06 (= d90c0a1): r16 a32 dropout0.05 in_proj,x_proj; lr 8e-5 cosine warmup0.03 wd0.01; bs32 ga8; 15 epochs; no grad ckpt; max_len 1024; lm_eval_p1-5 CYCLE; keep-best eval_classification/all_accuracy patience 3; upstream masakhanews eng train/validation"
    args=("${common[@]}" finetune.training.gradient_accumulation_steps=16 "${keep_all[@]}")
    changes="save_total_limit 1->null; load_best_model_at_end true->false; early_stopping_patience 3->null; save_only_model false->true; report_to wandb->none; hub push off"
    deviations="1 GPU with gradient_accumulation_steps 8->16 to keep the inferred 2-GPU effective batch (bs32 x ga8 x 2 = 512), as for the hub intent adapters."
    ;;
  mamba2_news_mono_xho|mamba2_news_multi)
    code="$v15"
    [[ "$label" == mamba2_news_multi ]] && config=mamba_news_all || config=mamba_news_xho
    base=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-retained-bindings-20260916-v1/bases/mamba2
    base_file="$base/pytorch_model.bin"
    tok="$base"
    tok_sha=3be3a5fda9551681d05a215392a292cff148b132fb9a5a7c693f278d37f20d13
    launcher=python
    original="Kombuys /scratch/alombard/sallm/results/full_matrix_targeted_recovery_20260922_kombuys_v15/train/$label (v15 snapshot, arm runner run_v15_kombuys_recovery_arm_20260922_v2.sh, RTX 5090; kept epoch 1 of 15 by eval_classification/all_f1, early-stopped after epoch 4)"
    recipe_note="$config at v15: r16 a32 dropout0 in_proj,x_proj; lr 8e-5 cosine warmup0.03 wd0.01; bs2 ga8 eval bs1; 15 epochs; bf16; grad ckpt; max_len 2048; lm_eval_p1-5 CYCLE; seed 42; base and tokenizer = mamba2 base dir (weights 7eaa6b8a, identical to the Kombuys base)"
    args=(
      "finetune.model.init_checkpoint=$base"
      "finetune.tokenizer.path=$base"
      "finetune.training.run_name=$label-targeted-recovery-v15-kombuys"
      ++finetune.training.seed=42
      "${common[@]}"
      "finetune.peft.kwargs.target_modules=[in_proj,x_proj]"
      finetune.training.per_device_train_batch_size=2
      finetune.training.per_device_eval_batch_size=1
      finetune.training.gradient_accumulation_steps=8
      "${keep_all[@]}"
    )
    env_extra=(
      "SALLM_PRELOAD_RUNNER=$v15/.audit/run_train_validation_only_20260914.py"
      SALLM_PRELOAD_FN=_install_loader
      SALLM_T2X_TRAIN_VALIDATION_ONLY=1
      "SALLM_T2X_CACHE_DIR=$root/tmp/mm_recipe/t2x_unused"
      SALLM_MAMBA_VALIDATION_LABEL_MICROBATCH=17
    )
    changes="save_total_limit 1->null; load_best_model_at_end true->false; early_stopping_patience 3->null; save_only_model false->true"
    deviations="HEX L40S instead of Kombuys RTX 5090; HEX live venv (torch 2.9.1, transformers 4.57.3, trl 0.26.2, peft 0.18.1, mamba_ssm 2.3.2.post1, datasets 4.8.5) instead of the Kombuys venv (same versions except datasets 3.6.0); masakhanews read offline from the HEX HF cache at the v15-pinned revision fa3b5fff."
    ;;
  *) echo "ERROR: unknown label $label" >&2; exit 4 ;;
esac

# Author rule (25 Sep): a recipe with < 10 optimizer steps/epoch trains at effective batch 64 (same lr, epochs, schedule, LoRA).
declare -A eff64_ga=([mamba2_intent_mono_eng]=2 [mamba2_intent_mono_sot]=4 [mamba2_news_mono_eng]=2)
if [[ -n "${eff64_ga[$label]:-}" ]]; then
  new=()
  for a in "${args[@]}"; do [[ "$a" == finetune.training.gradient_accumulation_steps=* ]] || new+=("$a"); done
  args=("${new[@]}" "finetune.training.gradient_accumulation_steps=${eff64_ga[$label]}")
  deviations="$deviations RULE <10 steps/epoch -> effective batch 64: gradient_accumulation_steps=${eff64_ga[$label]} (per-device batch unchanged); supersedes the effective-batch-preserving GA above."
fi

run_label
