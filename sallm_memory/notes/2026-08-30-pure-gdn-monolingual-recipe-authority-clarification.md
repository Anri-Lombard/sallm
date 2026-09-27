# Pure-GDN Monolingual recipe authority clarification

Frozen at 17:35 SAST on 30 August 2026, before any Monolingual adapter was
submitted and without consulting any adapter held-out result.

This clarification preserves the 9 August correction preregistration. It does
not open a new search or transfer the enhanced Multilingual winner wholesale.
That preregistration says each applicable Monolingual adapter is trained and
validation-selected at its already frozen **family learning rate**. It also
fixes the remaining recipe at:

- architecture-complete LoRA targets `q_proj`, `k_proj`, `v_proj`, `a_proj`,
  `b_proj`, `g_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`;
- LoRA rank 16, alpha 32, and dropout 0.05;
- warmup ratio 0.03, AdamW betas 0.9/0.95, weight decay 0.01, cosine schedule,
  maximum gradient norm 1.0, BF16, assistant-only loss, and no packing.

The transferred learning rate is exactly the learning-rate component of the
final hashed family winner. There is no later choice between an earlier LR
screen and an enhanced winner. The winner's selected rank, alpha, dropout, or
warmup ratio does **not** transfer to a Monolingual adapter. Every Mono
execution manifest must bind that family LR and the fixed values above before
launch.

Exactly the 21 already listed Mono arms remain authorized: News 2, NER 3,
POS 3, SIB 6, Intent 4, T2X 1, and AfriHG 2. The frozen T2X HPO winner and its
selected checkpoint are the single T2X Xhosa Mono arm; they count toward 21,
and no second T2X training or duplicate Multilingual arm is authorized. No
other Mono submission is authorized before one verified eight-family freeze
manifest. Validation selects each checkpoint; official held-out tests remain
one-time and post-freeze only.

General reporting must also disclose that b1--b3 continuation authority was
added after their administrative wall-time terminations and after interim
validation artifacts existed. The b4--b7 rule was frozen after trial start but
before their terminal outcomes and activates only after the same wall-time
termination. These are metric-independent amendments: eligibility, checkpoint,
and order do not depend on metric magnitude. They preserve the same trial
state and candidate order, but must not be described as unamended
preregistered runs.
