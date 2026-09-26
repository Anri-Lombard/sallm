# Pure-GDN News official terminal runtime failure

Official News bundle `1290454` failed `1:0` on 3 September 2026 after opening
the first English test task but before evaluating examples, writing predictions,
or producing a result file. The immutable runner created an lm-eval task-path
shim through the canonical scratch location of the symlinked virtual
environment, while lm-eval compared it with the unresolved home-directory
location. Its diagnostic `relative_to` call raised `ValueError` before task
evaluation.

The News result root contains zero files and no prediction, summary, or metric
exists. Nevertheless, the frozen one-time rule became terminal when the News
held-out task payload opened. Preserve job `1290454`, its log, the `news_v2`
protocol, and snapshot. Never correct or retry current-adapter News official
testing. News Mono/Multi official results remain terminally missing.
