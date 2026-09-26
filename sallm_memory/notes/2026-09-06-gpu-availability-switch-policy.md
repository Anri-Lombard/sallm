# Availability-aware A100 scheduling

On 6 September the user asked to switch from 80GB to 40GB and update the
scheduled monitor to switch to the most available GPUs as needed if viable.
This authorizes conditional A100-family scheduling changes, not scientific
recipe changes or interruption without a demonstrated overall benefit.

Live check: six physically free 40GB cards (four amperemk, two ampere) and
one free 80GB card. Physical availability does not prove account/QOS access.
B5 1312084 was at step 12027; b6 1312231 at 11880. Both latest complete
saved checkpoints remain 10912. Immediate cancellation would discard roughly
two hours of b5 work and nearly two hours of b6 work. With a free 80GB card
for b3, no net switching benefit is demonstrated. Keep both jobs running.

At each monitor, compare expected completion of the remaining authorized
work using live queues/account caps, measured throughput, memory fit,
migration overhead and unsaved work. Prefer switches between completed jobs
or at an already authorized complete-state boundary. Use only one A100
memory family at a time, including schedulable pending jobs. Before switching,
clear or safely dependency-hold owned jobs from the old family. Preserve
the four-owned-job ceiling and any stricter account/protocol limit.

Before 40GB execution, verify the existing equivalence evidence covers the
exact device/runtime/recipe; do not assume amperemk and ampere interchangeable.
Freeze a prospective hardware/resource amendment binding unchanged scientific
code, data, state and candidate wrapper. Existing terminal/one-continuation
limits remain. Ask for direction if switching requires an extra recovery or
new scientific gate beyond current authority; never rerun a failed gate.
No L40S, GDN on Kombuys, GPU-family overlap, recipe/seed/batch changes,
held-out-based choices, or changes to other users' jobs.

The monitor should execute a viable authorized switch; otherwise retain
useful running work and record why. No running job was changed here. B3
preflight 1312488 has passed; its existing 80GB continuation remains eligible
after the already required final checks.
