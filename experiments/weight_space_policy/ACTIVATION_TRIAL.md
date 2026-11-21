# First direct-edit trial

The fixed Gemma 4 E2B checkpoint and L40S runner from the baseline are reused. The five baseline requests are development data. The published baseline's current-policy condition returned the cap on every request, including a valid request below it. The teacher prompt in this trial explicitly instructs the model to preserve valid requests.

One candidate is derived before held-out inference. Read the last prompt token at layer 17's down projection for each development request with and without the teacher prompt. Use the mean difference in projection outputs and the mean unprompted input to calculate a half-strength outer product on the 128 largest output and 32 largest input coordinates. Cap requested deltas at 0.04, discard BF16 no-ops, and freeze the manifest digest. No gradients, training, or holdout-based candidate selection.

The held-out set has 30 requests across five graph-derived baselines, under a fixed 5% policy. Six conditions compare the original checkpoint with no policy, old policy, or explicit current policy against the edited checkpoint with no policy or old policy and an equally sized random edit with no policy. The seeded control relocates each delta to a disjoint cell with identical original BF16 bits. Abort if effective changes or L2 do not match, or if the projection does not restore byte-for-byte.

The primary score is the number of exact proposals, using the original request when eligible and flooring the policy cap to one decimal place when it is exceeded. Report authorized-distance preservation and clamping separately, together with independent PySHACL verdicts. In each no-policy arm, also check five explicit policy changes to 2% or 12% and four unrelated short-answer prompts. Treat the pilot as promising only if the edited no-policy arm achieves at least three more exact requests than both the original and random no-policy arms, preserves at least as many valid requests as the original, and preserves its exact unrelated and explicit-policy-change scores. One candidate and one seeded negative control give exploratory evidence. A null cannot rule out other weight edits.

The independent gate derives runner history from synthetic RDF observations and remains the authority for policy checks. The trial does not write checkpoints or alter production behavior.

From the committed checkout with a Modal profile, use the existing launcher:

```sh
python -m experiments.weight_space_policy.modal_baseline --trial
python -m experiments.weight_space_policy.modal_baseline --trial --launch --remaining-usd <authorized-remaining-budget>
```
