"""Decode the distance under matched attention masks after an unrestricted prefill."""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .distance_diagnostic import HISTORIES, JSON_SYSTEM, REVISION, cases, prompt

ARMS = ("unmasked", "cap_digits", "control_words")


@torch.inference_mode()
def run(checkpoint, output):
    torch.set_num_threads(6)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16).eval()
    probes = {p.id: p for p in cases(HISTORIES, "diagnostic")}
    previous = Path(__file__).parent / "results/pre_number/rows.json"
    results = []
    for row in json.loads(previous.read_text())["rows"]:
        policy = prompt(probes[row["id"]], "rule_and_cap")
        chat = tokenizer.apply_chat_template(
            [{"role": "system", "content": JSON_SYSTEM}, {"role": "user", "content": policy}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        text = chat + row["prefix"]
        phrase = "For this runner the cap is "
        assert text.count(phrase) == 1
        start = text.index(phrase)
        cap_start = start + len(phrase)
        assert text[cap_start:].startswith(row["target"] + " km.")
        tokens = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        digits, controls = [], []
        for i, (a, b) in enumerate(tokens.offset_mapping):
            token = text[a:b].strip()
            if cap_start <= a < cap_start + len(row["target"]) and token.isdigit():
                digits.append(i)
            if start <= b-1 < start+len("For this runner") and token:
                controls.append(i)
        assert len(digits) == len(controls) == 3
        ids = torch.tensor([tokens.input_ids]).repeat(len(ARMS), 1)
        length = ids.shape[1]
        cache = model(ids[:, :-1], use_cache=True, logits_to_keep=1).past_key_values
        current = ids[:, -1:]
        mask = torch.ones_like(ids)
        mask[1, digits] = 0
        mask[2, controls] = 0
        generated = []
        for step in range(8):
            result = model(current, past_key_values=cache, use_cache=True, attention_mask=mask,
                           position_ids=torch.full_like(current, length-1+step))
            cache = result.past_key_values
            current = result.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(current)
            mask = torch.cat([mask, torch.ones_like(current)], dim=1)
        suffixes = tokenizer.batch_decode(torch.cat(generated, dim=1))
        arms = {}
        for arm, raw in zip(ARMS, suffixes):
            match = re.match(r'(\d+(?:\.\d+)?)(?=["\s,}])', raw)
            value = match[1] if match else None
            arms[arm] = dict(raw=raw, value=value, exact=value == row["target"])
        results.append(dict(id=row["id"], target=row["target"], prefix=row["prefix"],
                            cap_positions=digits, control_positions=controls, arms=arms))
        output.write_text(json.dumps(dict(revision=REVISION, rows=results), indent=2)+"\n")
        print(row["id"], [arms[key]["value"] for key in ARMS], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
