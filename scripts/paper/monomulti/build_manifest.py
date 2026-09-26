import json, csv
S='/private/tmp/claude-501/-Users-anrilombard-Desktop-sa-architecture-comparison-paper/8c159bcd-58cf-4f3d-afb4-5e7b76e82930/scratchpad/monomulti'
probe={}
for f,host in (('probe/hex_probe.json','hex'),('probe/kom_probe.json','kombuys'),('probe/hex2_probe.json','hex')):
    for r in json.load(open(f'{S}/{f}')):
        probe.setdefault(r['key'],[]).append((host,r))
sheet={(r['model'],r['task'],r['language'],r['regime']):r for r in json.load(open(f'{S}/sheet_rows.json'))}
AM={'mzansilm':'MzansiLM','mamba2':'Mamba','xlstm':'xLSTM','gdn':'GDN'}
TM={'news':'News','sib':'SIB-200','intent':'Intent','ner':'NER','pos':'POS'}
LANGS={'news':['eng','xho'],'sib':['afr','eng','nso','sot','xho','zul'],'intent':['eng','sot','xho','zul'],'ner':['tsn','xho','zul'],'pos':['tsn','xho','zul']}
# per (arch,regime,task[,lang]) decisions: (primary probe key, status, provenance, note)
D={}
def d(a,reg,t,lang,status,prov,note,key=None,alt=''):
    D[(a,reg,t,lang)]=dict(status=status,prov=prov,note=note,key=key,alt=alt)
H='/scratch/lmbanr001/masters/sallm'
for l in LANGS['news']+LANGS['sib']+LANGS['intent']+LANGS['ner']+LANGS['pos']: pass
for t in TM:
    for l in LANGS[t]:
        st='OK'; note=''
        if t in('ner','pos'):
            note='already scored with full-matrix sequence protocol (unit %s)'%({'ner':{'tsn':22,'xho':23,'zul':24},'pos':{'tsn':27,'xho':28,'zul':29}}[t][l])
        d('gdn','Mono',t,l,st,'pure_gdn_mono_familywise_v1 seed 42: GDN per-family HPO-selected recipe (adapter_hpo_v3) applied per language; best-on-validation checkpoint',note)
    d('gdn','Multi',t,'*','OK',{'news':'adapter_hpo_v3 news stage_a/a0 (env_correction rerun)','sib':'adapter_hpo_v3 sib stage_a/a0','intent':'adapter_hpo_v3 intent stage_a/a1','ner':'adapter_hpo_v3 ner stage_b/b7 (r32)','pos':'adapter_hpo_v3 pos stage_a/a2'}[t]+'; HPO-selected arm',{'ner':'already scored with full-matrix sequence protocol (unit 78)','pos':'already scored with full-matrix sequence protocol (unit 80)'}.get(t,''))
# xLSTM
for l in ['eng','xho']: d('xlstm','Mono','news',l,'OK','xlstm_news_recovery_20260728 %s_rolefix_val_r2 (chat-role fix retrain, validation-selected lr); r16'%l,'')
d('xlstm','Multi','news','*','OK','xlstm_news_recovery_20260728 all_rolefix_val_r2; r128','')
for l in LANGS['sib']: d('xlstm','Mono','sib',l,'OK','Kombuys standardized_adapter_recovery_20260915_jbuys_xlstm_sib_mono_%s (reconstruction of sheet recipe: r256+embeddings, lr 9.23e-5, 10 ep, bs16x2)'%('v5' if l=='afr' else 'v6'),'Kombuys-only; copy to HEX or score on Kombuys')
d('xlstm','Multi','sib','*','SUSPECT','ft_xlstm_125m_sib_all/k6ob2x17 (W&B sweep run, r256+embeddings)','adapter_config base is anrilombard/sallm-xlstm-125m (identity with native-3epoch unverified); full-matrix binding was ambiguous among 3 candidates; only this one still exists (downstream pad64 and hpo_recovery a0_pad64 dirs are gone); no record that k6ob2x17 was the validation-selected sweep run',key='xlstm|Multi|sib|*cand_k6ob2x17')
for l in LANGS['intent']: d('xlstm','Mono','intent',l,'OK','intent_recovery_20260729 xlstm_mono_%s_r4_r2 (clean-data retrain, r4)'%l,'')
d('xlstm','Multi','intent','*','OK','intent_recovery_20260729 xlstm_clean_r4 checkpoint-442 (best on validation; final_adapter is the same run)','',key='xlstm|Multi|intent|*')
for l in LANGS['ner']: d('xlstm','Mono','ner',l,'OK','historical_selected_adapter_recovery_20260917_hex_v3 (re-materialized historical W&B-selected recipe: lr 4.50e-4, r16, 20 ep, bs4x16)','original weights (xlstm_downstream_3epoch_20260602_pad64) are gone; this is a retrain from the frozen recipe, not byte-identical')
d('xlstm','Multi','ner','*','SUSPECT','ft_xlstm_125m_ner_all/4vfuoadl (W&B sweep run of 5; r128+embeddings)','sheet cell says "(HPO-selected test)": the sweep run appears to have been picked on TEST; adapter_config base is anrilombard/sallm-xlstm-125m (not the native-3epoch base used elsewhere; identity unverified); chat template df244b52 differs from the others')
for l in LANGS['pos']: d('xlstm','Mono','pos',l,'OK','Kombuys standardized_adapter_recovery_20260915_jbuys_pos_t2x_v2 (r128+embeddings, per-language lr, 12 ep, bs8x4)','Kombuys-only')
d('xlstm','Multi','pos','*','OK','Kombuys standardized_adapter_recovery_20260915_jbuys_pos_t2x_v2 xlstm_pos_multi','Kombuys-only')
# Mamba
d('mamba2','Mono','news','eng','OK','hub anrilombard/sallm-mamba2-masakhane-masakhanews-eng@3ea3e4f (copy in full-matrix-retained-bindings snapshot)','')
d('mamba2','Mono','news','xho','OK','Kombuys full_matrix_targeted_recovery_20260922_kombuys_v15 mamba2_news_mono_xho (default mamba_news_xho config, best=epoch 1/15)','hub original sallm-mamba2-masakhane-masakhanews-xho@967337a exists but not downloadable on remotes (no HF token)',alt='hub:anrilombard/sallm-mamba2-masakhane-masakhanews-xho@967337a')
d('mamba2','Multi','news','*','OK','Kombuys v15 mamba2_news_multi (default mamba_news_all, best=epoch 1/15)','hub original sallm-mamba2-masakhane-masakhanews-eng-xho@46c1ed6 not on remotes',alt='hub:anrilombard/sallm-mamba2-masakhane-masakhanews-eng-xho@46c1ed6')
for l in LANGS['sib']: d('mamba2','Mono','sib',l,'MISSING','hub only: anrilombard/sallm-mamba-sib_%s (Jan, r256+embeddings) vs anrilombard/sallm-mamba2-davlan-sib200-%s_latn (Feb, r16 in_proj/x_proj)'%(l,l),'no copy on HEX/Kombuys; remotes have no HF token; Kombuys v15 recovery arms 10-15 never ran (loop died in arm 9 on 22 Sep)')
d('mamba2','Multi','sib','*','MISSING','hub only: anrilombard/sallm-mamba-sib_all@99871c4 vs sallm-mamba2-davlan-sib200-afr_latn-...-zul_latn','no copy on remotes; v15 arm 16 never ran')
for l in LANGS['intent']: d('mamba2','Mono','intent',l,'OK','hub anrilombard/sallm-mamba2-masakhane-injongointent-%s (copy in retained-bindings snapshot + Kombuys retained_standardized)'%l,'')
d('mamba2','Multi','intent','*','OK','intent_recovery_20260729 mamba_clean_r2 checkpoint-888 (best on validation)','',key='mamba2|Multi|intent|*')
for l in LANGS['ner']: d('mamba2','Mono','ner',l,'SUSPECT','v9 targeted recovery mamba2_ner_mono_%s (default mamba_ner_%s: lr 8e-5, bs2 x GA%s = eff. 256, 15 ep)'%(l,l,{'tsn':128,'xho':128,'zul':64}[l]),'only %s optimizer steps/epoch; best checkpoint = epoch 1 (step %s), early-stopped; effectively untrained. Hub alternatives (sallm-mamba-ner_%s, sallm-mamba2-masakhaner2-%s) not on remotes'%({'tsn':6,'xho':6,'zul':12}[l],{'tsn':6,'xho':6,'zul':12}[l],l,l))
d('mamba2','Multi','ner','*','OK','v9 targeted recovery mamba2_ner_multi (default mamba_ner_all, 680 steps, 10 ep)','')
for l in LANGS['pos']: d('mamba2','Mono','pos',l,'OK','Kombuys v15 mamba2_pos_mono_%s (default mamba_pos_%s)'%(l,l),'Kombuys-only; hub original sallm-mamba2-masakhane-masakhapos-%s not on remotes'%l)
d('mamba2','Multi','pos','*','MISSING','Kombuys v15 mamba2_pos_multi: training died at step ~389/540 (no final_adapter); checkpoint-324 (best so far) and checkpoint-360 exist','hub original sallm-mamba2-masakhane-masakhapos-tsn-xho-zul@92eb336 not on remotes; old ft_mamba_125m_pos_all_tagseq uses a different (tag-sequence) format',key='mamba2|Multi|pos|*ckpt360')
# MzansiLM
for l in ['eng','xho']: d('mzansilm','Mono','news',l,'OK','historical_selected_adapter_recovery_20260917_hex_v3 (historical recipe: r128+embed/lm_head, lr 3e-5, 20 ep, bs4x4x2GPU, v5 training templates)','hub original sallm-llama-masakhane-masakhanews-%s no longer exists; retrain not byte-identical'%l)
d('mzansilm','Multi','news','*','OK','hex_v3 mzansilm_news_multi (15 ep)','hub original deleted')
for l in LANGS['sib']: d('mzansilm','Mono','sib',l,'OK','sallm_recovery/mzansilm_sib_mono_20260916_v4r2 (llama_sib_%s recipe: r128+embed/lm_head, lr 3e-5, 15 ep)'%l,'hub original deleted')
d('mzansilm','Multi','sib','*','OK','hex_v3 mzansilm_sib_multi (15 ep, v1+v5 templates)','hub original deleted')
for l in LANGS['intent']: d('mzansilm','Mono','intent',l,'OK','hub anrilombard/sallm-llama-masakhane-injongointent-%s (copy in retained-bindings snapshot + Kombuys)'%l,'adapter_config.base_model_name_or_path says anrilombard/sallm-mamba-125m (metadata slip; q_proj/v_proj LoRA on MzansiLM); pass the MzansiLM base explicitly')
d('mzansilm','Multi','intent','*','OK','intent_recovery_20260729 llama_clean_r2 checkpoint-1764 (best on validation)','Kombuys control run in data/intent-position-control.csv used this adapter (train-position F1 28.4/14.3/...)',key='mzansilm|Multi|intent|*')
for l in LANGS['ner']: d('mzansilm','Mono','ner',l,'OK','hex_v3 mzansilm_ner_mono_%s (historical recipe r128+embed, %s ep)'%(l,{'tsn':20,'xho':50,'zul':20}[l]),'hub original deleted')
d('mzansilm','Multi','ner','*','OK','v9 targeted recovery mzansilm_ner_multi (default llama_ner_all: r16 q/v, lr 3e-5, 10 ep)','hub sallm-llama-masakhane-masakhaner2-tsn-xho-zul still exists (not on remotes)')
d('mzansilm','Mono','pos','tsn','SUSPECT','standardized_adapter_recovery_20260914_hex_v1 mzansilm_pos_mono_tsn (lr 1.6e-4, 10 ep, bs2x4; seal_status=unsealed)','recipe differs from xho/zul (lr 3e-5, 20 ep) and from the other Mono POS adapters; never sealed')
for l in ['xho','zul']: d('mzansilm','Mono','pos',l,'OK','v9 targeted recovery mzansilm_pos_mono_%s (default llama_pos_%s: r16, lr 3e-5, 20 ep)'%(l,l),'hub original sallm-llama-masakhane-masakhapos-%s exists (not on remotes)'%l)
d('mzansilm','Multi','pos','*','OK','v9 targeted recovery mzansilm_pos_multi (llama_pos_all r16, 20 ep)','hub original sallm-llama-masakhane-masakhapos-tsn-xho-zul@c707998 exists')
REUSE={('gdn','Mono','ner'):{'tsn':'67.83 (u22)','xho':'52.68 (u23)','zul':'50.26 (u24)'},('gdn','Multi','ner'):{'tsn':'78.35 (u78)','xho':'70.00 (u78)','zul':'73.61 (u78)'},
 ('gdn','Mono','pos'):{'tsn':'85.08 (u27)','xho':'85.63 (u28)','zul':'87.57 (u29)'},('gdn','Multi','pos'):{'tsn':'85.64 (u80)','xho':'83.44 (u80)','zul':'86.01 (u80)'}}
HOST={'news':'kombuys','sib':'kombuys','intent':'kombuys','ner':'hex','pos':'hex'}
def action(a,reg,t,l,status):
    if (a,reg,t) in REUSE: return 'REUSE full-matrix General-protocol score '+REUSE[(a,reg,t)][l]
    if status=='MISSING': return 'RETRAIN (or fetch hub original, see plan.md) then rescore on '+HOST[t]
    if status=='SUSPECT': return 'RESCORE as-is on '+HOST[t]+' AND retrain replacement (see plan.md)'
    return 'RESCORE on '+HOST[t]
rows=[]
for a in AM:
  for reg in ('Mono','Multi'):
    for t in TM:
      for l in LANGS[t]:
        dd=D[(a,reg,t,l if reg=='Mono' else '*')]
        key=dd['key'] or f"{a}|{reg}|{t}|{l if reg=='Mono' else '*'}"
        pr=probe.get(key,[])
        paths=[];trees=[];wsha=[]
        for host,r in pr:
            if r.get('exists'):
                paths.append(f"{host}:{r['path']}"); trees.append(r.get('tree_sha256',''))
                wsha.append(';'.join(f"{n}={v[1]}" for n,v in r.get('weights',{}).items()))
        # extra copies
        for k2,v2 in probe.items():
            if k2!=key and k2.startswith(f"{a}|{reg}|{t}|{l if reg=='Mono' else '*'}") and 'alt' not in k2 and 'cand' not in k2:
                for host,r in v2:
                    if r.get('exists'): paths.append(f"{host}:{r['path']}"); trees.append(r.get('tree_sha256','')); wsha.append(';'.join(f"{n}={v[1]}" for n,v in r.get('weights',{}).items()))
        lora=next((r.get('lora') for h,r in pr if r.get('lora')),{}) or {}
        cfg=next((r.get('cfg') for h,r in pr if r.get('cfg')),{}) or {}
        ts=next((r.get('trainer_state') for h,r in pr if r.get('trainer_state')),{}) or {}
        sc=sheet.get((AM[a],TM[t],l,reg),{})
        status=dd['status']
        if status!='MISSING' and not paths: status='MISSING'
        eff=None
        try: eff=int(cfg.get('per_device_train_batch_size'))*int(cfg.get('gradient_accumulation_steps'))
        except Exception: pass
        extra=[]
        if a=='gdn' and t in ('sib','intent'): extra.append('validation metric collapsed for SIB/Intent (SIB macro-F1 0.0576 for every arm), so keep-best kept the epoch-1 weights')
        if a=='mamba2' and t=='news' and 'kombuys_v15' in ' '.join(paths): extra.append('v15 retrain kept epoch 1 of 15 (validation F1 0.21/0.24)')
        if a=='mzansilm' and t in ('news','ner','sib') and not (t=='ner' and reg=='Multi'): extra.append('trained on the v5/v1 training templates (r128 + embed/lm_head), not the lm_eval_p* prompts it is scored with')
        if extra: dd=dict(dd, note=(dd['note']+'; ' if dd['note'] else '')+'; '.join(extra))
        rows.append(dict(architecture=a,regime=reg,task=t,language=l,status=status,
            adapter_paths=' | '.join(paths),alt_candidates=dd['alt'],tree_sha256=' | '.join(dict.fromkeys(trees)),adapter_weight_sha256=' | '.join(dict.fromkeys(wsha)),
            lora=f"r={lora.get('r')} alpha={lora.get('lora_alpha')} targets={','.join(sorted(lora.get('target_modules') or []))} modules_to_save={lora.get('modules_to_save')}" if lora else '',
            train_cfg=(f"lr={cfg.get('learning_rate')} epochs={cfg.get('num_train_epochs')} bs={cfg.get('per_device_train_batch_size')} ga={cfg.get('gradient_accumulation_steps')} eff_bs_per_device={eff} sched={cfg.get('lr_scheduler_type')}" if cfg else ''),
            selected_checkpoint=(f"step={ts.get('global_step')} epoch={ts.get('epoch')} of {ts.get('num_train_epochs')}" if ts else ''),
            provenance=dd['prov'],notes=dd['note'],
            action=action(a,reg,t,l,status),
            sheet_score=sc.get('score',''),sheet_cell=f"{sc.get('source_sheet','')}!{sc.get('source_cell','')}" if sc else ''))
with open(f'{S}/manifest.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
from collections import Counter
print(len(rows),Counter(r['status'] for r in rows))
print(Counter((r['regime'],r['status']) for r in rows))
