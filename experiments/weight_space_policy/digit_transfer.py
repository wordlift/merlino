"""Patch one residual state before a distance digit."""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .distance_diagnostic import HISTORIES, JSON_SYSTEM, REVISION, cases, prompt

LAYERS = (7, 15, 23, 27)
FIELD = re.compile(r'"planned_km"\s*:\s*"?(\d+\.\d+)')


@torch.inference_mode()
def run(checkpoint, output, before_number=False):
    torch.set_num_threads(6)
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16).eval()
    probes = {p.id: p for p in cases(HISTORIES, "diagnostic")}
    directory = Path(__file__).parent / "results/distance_diagnostic"
    rows = json.loads((directory / "rule.json").read_text())["rows"]
    results = []
    for row in rows:
        if not row["oversized"]:
            continue
        p = probes[row["id"]]
        field = FIELD.search(row["raw"])
        actual, target = field[1], row["cap"]
        pos = next(i for i,(a,b) in enumerate(zip(actual,target)) if a != b)
        start = 0 if before_number else pos
        prefix = row["raw"][:field.start(1)+start]
        # Change the conflicting digit only, preserving the shared prefix.
        digit = int(target[pos])
        foil = target[:pos] + str(digit-1 if digit else 1) + target[pos+1:]
        def inputs(policy):
            chat = tok.apply_chat_template([dict(role="system",content=JSON_SYSTEM),
                dict(role="user",content=policy)], tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            return tok(chat+prefix,add_special_tokens=False,return_tensors="pt").input_ids
        recipient = inputs(prompt(p,"rule"))
        supplied = prompt(p,"rule_and_cap")
        false_prompt = supplied.replace(f"cap is {target} km.", f"cap is {foil} km.")
        assert false_prompt != supplied
        def generate_suffixes(ids):
            tokens = model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,
                                    max_new_tokens=8 if before_number else 5,pad_token_id=tok.eos_token_id)
            return tok.batch_decode(tokens[:, ids.shape[1]:])
        def continuation(ids):
            return generate_suffixes(ids)[0]
        base = continuation(recipient)
        record = dict(id=p.id,target=target,foil=foil,prefix=prefix,base=base,arms=[])
        for label,policy in (("true",supplied),("foil",false_prompt)):
            donor = inputs(policy)
            record[label+"_donor"] = continuation(donor)
            states = {}
            def capture(layer):
                def save(module, args, out):
                    states[layer] = out[:, -1].clone()
                return save
            hooks = [model.model.layers[layer].register_forward_hook(capture(layer))
                     for layer in LAYERS]
            try:
                model(donor, use_cache=False)
            finally:
                for hook in hooks:
                    hook.remove()
            # Each row receives exactly one transplant, at its assigned layer.
            def patch_row(index, layer):
                def patch(module, args, out):
                    if out.shape[1] == recipient.shape[1]:
                        out[index, -1] = states[layer][0]
                return patch
            hooks = [model.model.layers[layer].register_forward_hook(patch_row(i, layer))
                     for i, layer in enumerate(LAYERS)]
            try:
                batch = recipient.repeat(len(LAYERS), 1)
                suffixes = generate_suffixes(batch)
            finally:
                for hook in hooks:
                    hook.remove()
            for layer, suffix in zip(LAYERS, suffixes):
                number = re.match(r'\d+(?:\.\d+)?',actual[:start]+suffix)
                record["arms"].append(dict(layer=layer,donor=label,suffix=suffix,
                    distance=number[0] if number else None))
        results.append(record)
        output.write_text(json.dumps(dict(revision=REVISION,before_number=before_number,rows=results),indent=2)+"\n")
        print(p.id,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output",type=Path)
    parser.add_argument("--before-number", action="store_true")
    args = parser.parse_args()
    run(args.checkpoint,args.output,args.before_number)
