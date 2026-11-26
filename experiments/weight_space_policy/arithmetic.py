"""Score a frozen Qwen checkpoint on cap-only prompts with unseen distances."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .discovery import RATIO, REVISION
from .trial import limit

HISTORIES = ("14", "16", "18", "22", "28", "36", "15", "17", "19", "21", "33", "47",
             "18.3", "21.7", "24.6", "28.9", "32.4", "35.8", "39.1", "42.7",
             "46.3", "49.8", "53.2", "56.5")
TEMPLATE = ("The weekly cap is 105% of last week's distance, rounded down to one decimal place. "
            "Last week's distance was {} km. What is the weekly cap in km? Reply with the number alone.")


@torch.inference_mode()
def run(checkpoint, output):
    torch.set_num_threads(6)
    tok = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16).eval()
    rows = []
    for i in range(0, len(HISTORIES), 8):
        histories = HISTORIES[i:i+8]
        prompts = [tok.apply_chat_template([dict(role="user", content=TEMPLATE.format(km))],
                   tokenize=False, add_generation_prompt=True, enable_thinking=False) for km in histories]
        inputs = tok(prompts, padding=True, add_special_tokens=False, return_tensors="pt")
        ids = model.generate(**inputs, do_sample=False, max_new_tokens=160, pad_token_id=tok.eos_token_id)
        answers = tok.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for km, raw in zip(histories, answers):
            expected = limit(Decimal(km), RATIO)
            bare = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*', raw)
            final_number = re.findall(r"\d+(?:\.\d+)?", raw)[-1]
            rows.append(dict(history=km, raw=raw, expected=str(expected),
                             final_number=final_number, numeric_exact=Decimal(final_number) == expected,
                             exact=bool(bare and Decimal(bare[1]) == expected)))
        output.write_text(json.dumps(dict(revision=REVISION, prompt=TEMPLATE, rows=rows), indent=2)+"\n")
        print(len(rows), sum(r["exact"] for r in rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
