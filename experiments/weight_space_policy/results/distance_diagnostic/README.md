# Numeric distance generation diagnostic

Use unedited Qwen3-1.7B in BF16 with greedy decoding on CPU. For each of six synthetic histories, test two eligible requests and two oversized requests. Four histories come from the decision experiment's development split. Two fractional histories exercise rounding down when ordinary nearest rounding would give a different cap. This is a paired development diagnostic, with the same case order, prompt wrapper, output contract, and generation settings across arms.

1. Original no-policy prompt.
2. Supply the 105% rule, floor to one decimal, and keep-or-cap instruction.
3. Use the identical rule prompt and append the computed cap.

Primary outcome is an exact distance accepted by the existing proposal parser. Diagnostic numeric accuracy also allows an otherwise valid JSON response with a numeric string in the distance field. Report eligible preservation, oversized clamping and independent graph acceptance separately. This design measures the effect of supplying information and does not directly localize a neural mechanism.
