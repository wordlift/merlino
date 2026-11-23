"""Patch one residual state immediately before the first incorrect distance digit."""
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
def run(checkpoint, output):
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
        prefix = row["raw"][:field.start(1)+pos]
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
        def continuation(ids):
            tokens = model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,
                                    max_new_tokens=5,pad_token_id=tok.eos_token_id)
            return tok.decode(tokens[0,ids.shape[1]:])
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
            for layer in LAYERS:
                block = model.model.layers[layer]
                state = states[layer]
                def patch(module,args,out):
                    if out.shape[1] == recipient.shape[1]:
                        out[:,-1] = state
                hook = block.register_forward_hook(patch)
                try: suffix = continuation(recipient)
                finally: hook.remove()
                number = re.match(r'\d+(?:\.\d+)?',actual[:pos]+suffix)
                record["arms"].append(dict(layer=layer,donor=label,suffix=suffix,
                    distance=number[0] if number else None))
        results.append(record)
        output.write_text(json.dumps(dict(revision=REVISION,rows=results),indent=2)+"\n")
        print(p.id,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    run(args.checkpoint,args.output)
