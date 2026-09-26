# General audit exporter correction

The user approved fixing the exporter and said routine execution repairs need
not wait for further permission. This authorizes the serialization-only repair
and a new metric-free audit, while preserving scientific gates and the original
failed job1318956, script, log and partial JSON. It does not authorize retrying
official tests after scientific output or changing a recipe or evaluator.

The exporter now encodes only YAML !function ScalarNodes as explicit tag/value
objects. All other unsupported objects still fail closed. A runnable check
verifies exact tag/value roundtrip and rejection of unknown objects. Task
definitions, task counts, split checks and generation checks are unchanged.
The new immutable script and output use v2 paths; old evidence is untouched.

Submitted CPU job1319707 once (maths/ada/normal,4CPUs,8GB,10minutes).
Sealed script SHA256: ac703a3c900395d591182abc5ba493bb6ecc3edde813e71d68b3a4220cec469e.
Remote script: /home/lmbanr001/masters/sallm_snapshots/general-confirm-20260907/general_task_contract_v2.py.
Log: /scratch/lmbanr001/masters/sallm/logs/jobs/gdn-general-task-contract-20260908-v2.out.
Output: /scratch/lmbanr001/masters/sallm/manifests/general-task-contract-20260908-v2.json.
The half-hour continuation is ACTIVE again and records the user's repair
authority. Await terminal validation; submission alone is not audit acceptance.
