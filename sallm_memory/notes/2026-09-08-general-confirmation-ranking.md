# General confirmation verification and ranking

CPU verifier 1317663 completed 0:0. All four retained checkpoint5456 to
final-adapter comparisons are exactly equal, each with 424 keys. B7 has
76,410,112 values; A2 has 71,762,560. The differing counts follow their frozen
rank32 versus rank16 recipes. All scheduled validation sidecars and exact
22,167-row coverage were previously verified. All four confirmations are
terminal-valid; no training rerun is required.

Run the unchanged frozen hpo_protocol.py with stage=confirm, direction=min,
using exactly b7/a2 at seeds13/42/87. Script SHA-256:
edf5638cae9d783b840088a16079770a907208d518602bad10e210829ed53568.
Registry SHA-256:
8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726.
Use source under general-b5-offline-20260906-v2, not the changed local script.
The 11 August preregistration selects by three-seed arithmetic mean and
freezes the winner's seed42 retained checkpoint, never its best individual
seed. Ranking remains validation-only. No General held-out access or Sheet
edit has occurred. Base acceptance provenance remains unresolved.

CPU ranking job 1318546 completed 0:0. Read-only ranking artifact:
/scratch/lmbanr001/masters/sallm/results/general-confirm-ranking-20260908.json
SHA-256: 1d51c1e3820f6a28e1c84969f39ed803b8854251924c58bc8d1710854a076cdf.
B7 mean=0.9156025847289161, sample SD=0.0019474801644413513;
A2 mean=0.9252625951023498, sample SD=0.003755238017431751.
Lower is better. The artifact retains all six validation values.

General freezes to B7 seed42 checkpoint5456 under the unchanged winner rule.
Root: /scratch/lmbanr001/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/general/stage_b/b7/seed_42
Retained adapter_model.bin SHA-256:
e1f2bb206603c6e3871d5e3ca8021eead7228a5946df5c448659ec2027c3b5f0.
Final adapter_model.safetensors SHA-256:
05dbf5fdf9721f0b2961d119cc46b6eb756598f4deda1e31e6a08c703a2b0015.
Recipe: LR=0.0001223079850011719, rank=32, alpha=64,
dropout=0.008055734634399415, warmup_ratio=0.0765941160917282.
General training and selection are complete. Next: bind the existing official
test evaluator, metric-free data/runtime preflight and exact General coverage,
then run each prescribed held-out arm once. Do not infer an arm list from
populated workbook rows or modify evaluation based on Mono/Multi outcomes.
