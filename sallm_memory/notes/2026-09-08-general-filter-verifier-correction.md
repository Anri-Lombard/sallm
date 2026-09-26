# General AfriMGSM verifier-only correction

At11:04 SAST group1 job1319958 failed1:0 after AfriMGSM evaluation finished.
The verifier assumed one sample record per document, but the frozen task
has two scoring filters. Saved artifact inspection shows250 original/effective
documents and500 sample records per task,250 unique document IDs per filter.
This is a verifier defect, not authorization to repeat inference.

Prospective correction: versioned verifier outside immutable source snapshot,
checking each declared filter independently for exact row count and IDs0..N-1.
Summary equality, exact task sets, n-samples and finite metrics remain required.
Regression self-check passes single/two-filter success, duplicate and missing
document rejection. Old verifier, failed job, predictions and scores preserved.
Incidental score text appeared in the failure-log tail; no score guides this
repair, scope, decoding or recipe. Do not release scores before full acceptance.

New verifier: /scratch/lmbanr001/masters/sallm/manifests/verify_general_pack_filters_20260908.py.
Run it on CPU against existing AfriMGSM output only. Never repeat AfriMGSM.
After success, g1 remainder BelebeleEng, BelebeleTso, AfriHGXho is eligible
only after fresh absent-claim/output/runtime checks and frozen binding checks.

CPU1321090 completed0:0, verified20tasks/5000documents against both filters.
Verifier SHA256 dad42fde3c18016d31fcce1dd25548a2ee284984d1d4129572a4458408e82489.
Artifact verified=true; result SHA8934576e54ce1234c4f1e06eae761542b5f4dd0b6b84ba458ea78389f6b59825.
All three remaining units have absent claims, output directories and runtime
directories. Fresh queue shows only g0/g2, both A10080. Submitted unchanged
frozen runner for only these three never-started units, A10080/8CPU/48hours,
preserving original1319958 and its two completed units. No inference repeated.
