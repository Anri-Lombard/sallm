set -euo pipefail
R=/scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924
S=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-retained-bindings-20260916-v1/bases/mamba2
D=$R/bases/mamba2_eosfix
mkdir -p $D
for f in config.json generation_config.json .gitattributes pytorch_model.bin README.md special_tokens_map.json tokenizer_config.json tokenizer.json; do cp $S/$f $D/$f; done
cd $D
PYTHONDONTWRITEBYTECODE=1 /home/lmbanr001/masters/sallm/.venv/bin/python - <<'PY'
import json
for f in ['config.json','generation_config.json']:
    d=json.load(open(f)); assert d['eos_token_id']==2 and d['pad_token_id']==1, d
    d['eos_token_id']=1; d['pad_token_id']=2
    open(f,'w').write(json.dumps(d,indent=2,sort_keys=True)+'\n')
    print(f, {k:d[k] for k in ['bos_token_id','eos_token_id','pad_token_id']})
PY
diff <(jq -S 'del(.eos_token_id,.pad_token_id)' $S/config.json) <(jq -S 'del(.eos_token_id,.pad_token_id)' config.json) && echo config_otherwise_identical
diff <(jq -S 'del(.eos_token_id,.pad_token_id)' $S/generation_config.json) <(jq -S 'del(.eos_token_id,.pad_token_id)' generation_config.json) && echo genconfig_otherwise_identical
sha256sum config.json generation_config.json pytorch_model.bin special_tokens_map.json tokenizer_config.json tokenizer.json README.md .gitattributes > $R/bases/mamba2_eosfix.FILES.sha256
cat $R/bases/mamba2_eosfix.FILES.sha256
echo TREE $(sort -k2 $R/bases/mamba2_eosfix.FILES.sha256 | sha256sum | cut -d' ' -f1)
chmod a-w $D/*
