"""Paired prompt diagnostic on unedited Qwen. Run with checkpoint and output directory."""
import argparse
import json
from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .discovery import REVISION, RATIO, cases
from .run import JSON_SYSTEM, extract_proposal
from .trial import limit, trial_prompt, validate

HISTORIES = ("26", "34", "44", "58", "31.6", "43.3")
CONDITIONS = ("none", "rule", "rule_and_cap")
NO_POLICY = "No numeric policy limit is provided in this prompt."
RULE = ("The weekly cap is 105% of last week's distance, rounded down to one decimal place. "
        "Keep the requested distance if it is within the cap. Otherwise return the cap.")


def prompt(probe, condition):
    base = trial_prompt(probe, "none")
    assert base.count(NO_POLICY) == 1
    if condition == "none":
        return base
    policy = RULE
    if condition == "rule_and_cap":
        policy += f" For this runner the cap is {limit(probe.baseline, RATIO)} km."
    return base.replace(NO_POLICY, policy)


def score(probe, raw):
    value, status = extract_proposal(raw)
    numeric = value
    if status == "not-number":
        body = raw.strip()
        if body.startswith("```json") and body.endswith("```"):
            body = body[7:-3].strip()
        obj = json.loads(body)
        field = obj["planned_km"]
        if isinstance(field, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", field):
            obj["planned_km"] = float(field)
            numeric, _ = extract_proposal(json.dumps(obj))
    cap = limit(probe.baseline, RATIO)
    return dict(id=probe.id, baseline=str(probe.baseline), request=str(probe.request),
                cap=str(cap), oversized=probe.request > cap, expected=probe.target,
                raw=raw, parse=status, parsed_km=value, diagnostic_km=numeric,
                exact=value == probe.target, numeric_exact=numeric == probe.target,
                accepted=validate(probe.baseline, RATIO, value)["accepted"])


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16).eval()
    probes = [p for i, p in enumerate(cases(HISTORIES, "diagnostic")) if i % 6 in (0, 1, 3, 5)]
    for condition in CONDITIONS:
        rows = []
        for start in range(0, len(probes), 8):
            batch = probes[start:start+8]
            texts = [tokenizer.apply_chat_template(
                [{"role":"system","content":JSON_SYSTEM},{"role":"user","content":prompt(p, condition)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in batch]
            inputs = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt")
            tokens = model.generate(**inputs, do_sample=False, max_new_tokens=160,
                                    pad_token_id=tokenizer.eos_token_id)
            raw = tokenizer.batch_decode(tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            rows.extend(score(p, s) for p,s in zip(batch, raw))
            record = dict(model="Qwen/Qwen3-1.7B", revision=REVISION, condition=condition, rows=rows)
            temp = args.output / f"{condition}.tmp"
            temp.write_text(json.dumps(record, indent=2) + "\n")
            temp.replace(args.output / f"{condition}.json")
            print(condition, len(rows), "strict", sum(r["exact"] for r in rows),
                  "numeric", sum(r["numeric_exact"] for r in rows), flush=True)


if __name__ == "__main__":
    main()
