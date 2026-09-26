#!/bin/bash
# Retrain a MzansiLM (LLaMA 125M) task adapter with its original recipe, changing only checkpoint
# handling: save every epoch (adapter only), no early stopping, no keep-best. The NER Multi label
# is new: Multi data trained with the MzansiLM Mono NER recipe. Evidence and per-label recipe
# notes are in $root/tmp/mm_recipe/RECIPES.md.
set -euo pipefail
umask 022
label="${1:?label}"
source /scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925/jobs/train_common.sh

base="$llama_base"
base_file="$llama_base/pytorch_model.bin"
base_sha="$llama_sha"
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
  mzansilm_intent_multi)
    code="$code_jul"
    config=llama_injongointent_all
    original="HEX /scratch/lmbanr001/masters/sallm/checkpoints/intent_recovery_20260729/llama_clean_r2 (Slurm 1130571, 2026-07-29, A100; kept checkpoint-1764 = epoch 4 of 8 by eval_classification/all_f1, early-stopped after epoch 7)"
    recipe_note="config defaults + launch overrides: r16 a32 dropout0.1 q_proj,v_proj; trainable_token_indices=3 added special tokens; lr 3e-5 cosine warmup0.1 wd0 beta2 0.95; bs4 ga4; 8 epochs (override of 20); bf16; grad ckpt (SFTConfig default); max_len 2048; lm_eval_p1-5 CYCLE train, ALL val; clean InjongoIntent split: train 7045, val 4155; seed 42, data_seed unset"
    args=(
      "${common[@]}"
      finetune.wandb.project=sallm-ft
      finetune.training.num_train_epochs=8
      ++finetune.training.metric_for_best_model=eval_classification/all_f1
      ++finetune.training.greater_is_better=true
      ++finetune.training.early_stopping_threshold=0.001
      "${keep_all[@]}"
    )
    env_extra=("SALLM_INJONGO_OFFLINE_SNAPSHOT=$injongo_snapshot")
    changes="save_total_limit 1->null; load_best_model_at_end true->false; early_stopping_patience 3->null (threshold 0.001 kept, inert); save_only_model false->true"
    deviations="code is a frozen copy of the live HEX repo taken 2026-09-25: configs, templates, data loaders and metrics are byte-identical to the July run; 16 source files changed later (FILES_CHANGED_AFTER_20260729T0904.txt), none on the LLaMA intent training path. InjongoIntent jsonl read from the local HF cache snapshot fe4be388. The original stopped after epoch 7 (early stopping); this run trains all 8 configured epochs."
    ;;
  mzansilm_intent_mono_eng|mzansilm_intent_mono_sot|mzansilm_intent_mono_xho|mzansilm_intent_mono_zul)
    lang="${label##*_}"
    code="$code_feb"
    config="llama_injongointent_$lang"
    original="hub anrilombard/sallm-llama-masakhane-injongointent-$lang (pushed 2026-02-17; copy /scratch/lmbanr001/masters/sallm_snapshots/full-matrix-retained-bindings-20260916-v1/adapters/mzansilm/intent/mono_$lang); no trainer state or log survives"
    recipe_note="per-cell config at git cf0ed06 (= d90c0a1): r16 a32 dropout0.1 q_proj,v_proj; lr 3e-5 cosine warmup0.1; bs4 ga4; 20 epochs; bf16; max_len 2048; lm_eval_p1-5 CYCLE; save_strategy no, i.e. the original kept the final epoch (20); upstream InjongoIntent train + upstream dev (eng: TEST via the Feb loader fallback)"
    args=("${common[@]}" finetune.training.gradient_accumulation_steps=8 "${keep_all[@]}")
    changes="save_strategy no->epoch; save_total_limit ->null; save_only_model ->true; load_best_model_at_end/early_stopping_patience set false/null (already off in the original); report_to wandb->none; hub push off"
    deviations="1 GPU with gradient_accumulation_steps 4->8: the Feb launcher (scripts/submit_llama_finetune_eval.sh -> scripts/launch_finetune.sh at cf0ed06, #SBATCH --gres=gpu:l40s:2) ran 2-process DDP, so this keeps the effective batch 32; inferred from the launcher, no log survives. Code state cf0ed06 vs the exact Feb working tree is unverifiable."
    if [[ "$lang" == eng ]]; then
      env_extra=(
        "SALLM_INJONGO_OFFLINE_SNAPSHOT=$injongo_snapshot"
        "SALLM_CLEAN_INJONGO_SPLIT_MODULE=$code_jul/src/main/sallm/data/loaders/injongointent_split.py"
      )
      deviations="$deviations DATA DEVIATION (eng only): the Feb loader had no eng dev split and fell back to TEST for validation, and Feb eng train overlaps test texts. This run keeps the Feb code and every Feb hyperparameter but replaces the InjongoIntent loader with the July clean pipeline (jobs/code/sallm-live-20260925 injongointent_split.py on the HF cache snapshot fe4be388): upstream eng train.jsonl minus rows whose normalized text appears in test.jsonl, then a stable md5 carve of 10% per intent as validation. Train 1046 rows, validation 111 rows (was: train 1779 upstream rows, validation = test 622 rows)."
    fi
    ;;
  mzansilm_ner_multi_monorecipe)
    hexv3_pkg=/scratch/lmbanr001/masters/sallm/recovery/historical_selected_adapter_recovery_20260917_v3
    code=/scratch/lmbanr001/masters/sallm_snapshots/pos-generation-recovery-20260915-v2-peft-embedding-fix
    sha_is "$code/SNAPSHOT_MANIFEST.sha256" e207d41d3d0ffc45669576db0704df460a239c88e32c50d60af422edadce50bc
    sha_is "$hexv3_pkg/PACKAGE_MANIFEST.sha256" ad787e957355d971e509817bfa628b4428e2a295e3231a623cd3e92555853018
    (cd "$hexv3_pkg" && sha256sum -c PACKAGE_MANIFEST.sha256 --quiet)
    sha_is "$std_venv/../runtime_manifest.json" 9f30eeb73d415bd8be3535b0f74689c5d53d04f622e0e3ddea04a8c5a2df24ab
    venv="$std_venv"
    config=llama_ner_all
    original="NEW adapter. Recipe = MzansiLM Mono NER reconstruction /scratch/lmbanr001/masters/sallm/results/historical_selected_adapter_recovery_20260917_hex_v3/train/mzansilm_ner_mono_{tsn,xho,zul} (runner $hexv3_pkg/run_historical_selected_train_validation_only_20260915.py, Slurm array 1344520, 2x L40S); data = llama_ner_all (the v9 mzansilm_ner_multi data: anrilombard/masakhaner-x-parquet@6aa65cdb, 1441 train rows per language, full dev)"
    recipe_note="r128 a256 dropout0.1 q_proj,v_proj + modules_to_save [embed_tokens,lm_head], trainable_token_indices null; lr 3e-5 cosine warmup0.1 wd0; bs4 eval bs4; effective batch 32; 20 epochs (tsn/zul value; xho used 50); bf16; max_len 2048; template masakhane_named_entity_recognition/v5 CYCLE; seed 42 data_seed 42; eval each epoch with the generation span-F1 callback; languages tsn+xho+zul: train 4323, val 2152"
    args=(
      "finetune.dataset.templates=[{id:masakhane_named_entity_recognition/v5,weight:1.0}]"
      finetune.dataset.template_choice=CYCLE
      finetune.dataset.max_seq_length=2048
      finetune.dataset.packing=false
      finetune.dataset.assistant_only_loss=true
      finetune.training.num_train_epochs=20
      finetune.training.eval_strategy=epoch
      finetune.training.dataloader_num_workers=2
      ++finetune.training.seed=42
      ++finetune.training.data_seed=42
      finetune.training.report_to=none
      finetune.hub.enabled=false
      finetune.hub.push_adapter=false
      finetune.hub.push_merged=false
      "finetune.training.run_name=$label-reconstruction"
      "finetune.model.init_checkpoint=$llama_base"
      "finetune.tokenizer.path=$sallm_tok"
      finetune.peft.kwargs.r=128
      finetune.peft.kwargs.lora_alpha=256
      finetune.peft.kwargs.lora_dropout=0.1
      "++finetune.peft.kwargs.target_modules=[q_proj,v_proj]"
      "++finetune.peft.kwargs.modules_to_save=[embed_tokens,lm_head]"
      ++finetune.peft.kwargs.trainable_token_indices=null
      finetune.training.learning_rate=0.00003
      ++finetune.training.weight_decay=0.0
      finetune.training.lr_scheduler_type=cosine
      finetune.training.warmup_ratio=0.1
      finetune.training.per_device_train_batch_size=4
      finetune.training.per_device_eval_batch_size=4
      finetune.training.gradient_accumulation_steps=8
      finetune.training.bf16=true
      "${keep_all[@]}"
    )
    env_extra=(
      "SALLM_HEXV3_RUNNER=$hexv3_pkg/run_historical_selected_train_validation_only_20260915.py"
      "SALLM_RECOVERY_MANIFEST=$hexv3_pkg/historical_selected_adapter_recovery_manifest_20260915_v2.json"
      EXPECTED_RECOVERY_MANIFEST_SHA256=cf6d02f233bb98060d87c85eb4163157b02765f96554e7ad0525c2545b875643
      SALLM_RECOVERY_DATA_ROOT=/scratch/lmbanr001/masters/sallm/results/historical_selected_adapter_recovery_20260917_hex_v3/assets/train_validation_only_final5
      "SALLM_RECOVERY_ADAPTER_ID=$label"
      SALLM_NER_VARIANT=xlstm
      "SALLM_HEXV3_EXTRA_RECORD={\"id\":\"$label\",\"architecture\":\"mzansilm\",\"task\":\"ner\",\"regime\":\"multi\",\"languages\":[\"tsn\",\"xho\",\"zul\"],\"config\":\"llama_ner_all\",\"epochs\":20,\"templates\":[\"masakhane_named_entity_recognition/v5\"],\"template_choice\":\"CYCLE\",\"expected_train_rows\":4323,\"expected_validation_rows\":2152}"
    )
    changes="relative to the Mono recipe: data = llama_ner_all languages and splits instead of one language; save_strategy no->epoch, save_total_limit ->null, save_only_model ->true (load_best/early stopping off, as in the Mono recipe)"
    deviations="Intended: Multi trained with the Mono recipe so MzansiLM uses one NER recipe across regimes (the earlier Multi used the repo default r16, lm_eval_p1-5, 10 epochs). Epochs = 20, the Mono value for tsn and zul (xho used 50). The Mono recipe ran on 2 GPUs (bs4 x ga4 x 2 = 32); this runs 1 GPU with ga 8 (bs4 x ga8 = 32). Data binding: SALLM_NER_VARIANT=xlstm selects the hex_v3 sealed copy of anrilombard/masakhaner-x-parquet@6aa65cdb (1441 train rows per language, full dev), the same files llama_ner_all loads; the Mono recipe itself trained on the full upstream CoNLL train files (3489/5718/5848 rows)."
    ;;
  *) echo "ERROR: unknown label $label" >&2; exit 4 ;;
esac

run_label
