# General resumption and Base coverage check

Date: 2026-09-04

The user explicitly instructed: "go ahead and start it", referring to General,
after being told that it was paused. This authorizes resuming the General
workstream. The completed Mono/Multi held-out results must not influence its
candidates, recipes, checkpoint choices, validation ranking, or confirmations.

No job was submitted in this pass. SSH through `hex`, `hex-direct`, and
`hex` via `jbuys-direct` all timed out before a remote command ran. Every
attempt placed `/scratch/slurm/bin/purequota` first in the remote command.

The active half-hour automation `gdn-general-continuation` now retries access
and continues this authorized work when its preflight requirements pass.

## Connection diagnosis

The user's subsequent instruction was to diagnose and fix the timeout.
`scutil --nc list` confirms the Mac's Tailscale VPN is connected.
`tailscale status` reports the configured `kombuys-hex-relay` peer at
`100.105.254.24` offline, last seen about 15 hours before this check, with no
received traffic. SSH verbose output stops at the TCP connection to that
peer's port 2222, before authentication or any HEX access. The configured
`hex` alias uses this peer through `ProxyJump jbuys`.

DNS resolves both campus endpoints, but bounded TCP probes to Kombuys
`137.158.60.140:22` and HEX `137.158.158.180:22` also time out. No configured
SSH route is currently reachable. The evidence establishes an unavailable
relay, but cannot distinguish host power/network loss from a stopped relay
or Tailscale service. There is no reason to alter working SSH keys or host
verification settings. Restoring the relay requires access to its host or
another functioning campus route; it cannot be restarted through the failed
connection. No SSH configuration or remote service was modified.

At the 20:54 UTC follow-up, direct HEX SSH still timed out. A saved Cisco
UCT VPN profile was found and its gateway was reachable. At the user's
explicit request, the saved Bitwarden HPC credentials were submitted through
the UCT Microsoft sign-in UI without exposing the password. Authentication
advanced to MFA, but VPN connectivity was not established. The user must
complete the mobile approval in Cisco before the direct campus route can be
verified. Do not repeatedly initiate sign-ins or MFA requests from monitoring.

On 5 September, a user-authorized fresh Cisco login reached SMS MFA. The user
supplied the code, which was submitted only in the UCT sign-in window. Cisco's
navigation log then recorded a redirect from Microsoft's SAS/ProcessAuth to
`https://vpn.uct.ac.za/+CSCOE+/message.html?mc=5` at 09:46:47 SAST. That
gateway endpoint returns `Validation failure.` (also observed on an independent
unauthenticated GET; this alone does not identify the failed validation).
Cisco remained disconnected and direct HEX SSH timed out. This establishes
that this VPN login did not complete; it does not establish a bad password,
bad MFA code, expired SAML session, or a HEX outage. iPhone Mirroring could
not connect, so no phone code was retrieved automatically. No secret or MFA
code is retained in these notes, and no certificate checks were relaxed.

Once connectivity returns, inspect owned jobs and exact artifact state before
any submission. Verify the completed b1/b2/b4 continuations and existing
terminal candidates. Resume the already authorized, still-unexecuted b5/b6
same-trial continuations only after archive, checkpoint-state, immutable-source,
offline-data, absent-final-output, and no-duplicate checks. Use the existing
candidate-specific recovery machinery and the ratified A100-80GB family.

The b3 continuation `1279472` failed after payload access on an SSL fetch.
Preserve that terminal failure. The user's resumption instruction permits
preparing a disclosed prospective infrastructure-recovery amendment, but the
old protocol must not be silently reinterpreted. Before a replacement can run,
bind its exact complete checkpoint and frozen training/validation data hashes,
prove offline availability, and record the narrow exception and isolated
execution provenance. Do not restart training from scratch, omit b3, or rank
an incomplete candidate set. This note is resumption authority and a handoff,
not a claim that a replacement has passed those checks.

After all eleven seed-42 candidates verify, use the frozen validation-only
ranking and four confirmations, including the already authorized 36-hour
confirmation limit. Freeze the General winner before its official tests.

Live Google Sheets readback of `GDN Results!A1:G43` found 21 populated Mono,
20 Multi, 41 Base, and zero General score cells. Mono/Multi coverage agrees
with the verified 3 September close-out. Base completeness is unresolved:
the 30 August corrected-base amendment quarantines 14 raw evaluation lanes,
but the later familywise note says Base is accepted 16/16, and the live Base
notes still cite August jobs. No corrected replacement or superseding
acceptance evidence was found in this bounded local review. Reconcile the
source artifacts before claiming Base is fully paper-ready. T2X/AfriHG were
explicitly excluded from the 14-lane correction. No Sheet value changed.
