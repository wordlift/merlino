# Merlino weight-edit results on Qwen3-1.7B

The frozen final-layer edit reproduced the two additional correct answers on the constrained keep-or-cap probe. The free-response task showed zero improvement. In six new matched-history pairs the edited model preferred keeping both requests under both answer mappings. This experiment provides no evidence that this edit transfers the 5% policy to Merlino proposals.

| Model | Keep eligible | Cap oversized | Exact JSON proposal | Numerically correct distance including strings | Oversized requests correctly capped in proposals |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 15/15 | 0/15 | 2/10 | 4/10 | 0/5 |
| Edited | 15/15 | 2/15 | 2/10 | 4/10 | 0/5 |
| Matched random edit | 15/15 | 0/15 | 2/10 | 4/10 | 0/5 |

On both changed decisions, the unrestricted top token was “Based”. Only the relative scores for A and B selected the correct action. Two improved cases and zero regressions on the original fixed probe give a two-sided exact paired sign test of 0.50. The separate explicit-rule baseline achieved 17/30, correctly capping nine oversized requests and keeping eight eligible requests.

Each new paired test holds the request and wording fixed and decreases last week's distance by 0.2 km. All three models got both decisions correct on zero of six pairs for either label assignment. The original probe used 30 examples, the free-response check used ten of those examples, and the new paired test used six independent histories. These small samples support a conclusion about this frozen patch and these prompts.

The source revision for this CPU rerun was b19bf22eb90f7923c29671e37611ad90dc646fc7. Input model revision and patch digests, raw proposals, margins, correct answers, and effective control magnitudes are in decision_results.json. Counts based on quoted numeric strings are diagnostic only. They do not change the strict JSON score or validator decisions. Both patches changed 4,096 BF16 weights at the same effective L2 norm. The rerun used no Modal compute.
