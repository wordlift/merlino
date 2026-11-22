"""Gradient-free Merlino decision probe. Run from the repository root."""

import argparse
from decimal import Decimal
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .patch import canonical_digest, temporary_patch
from .trial import (
    Probe,
    limit,
    trial_prompt,
    validate,
    restrict_to_matchable_cells,
    random_control,
)
from .run import JSON_SYSTEM, extract_proposal

REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
RATIO = Decimal("1.05")
DEV = (26, 34, 44, 58)
TEST = (23, 31, 41, 49, 57)


def cases(history, phase):
    result = []
    for i, baseline in enumerate(history):
        baseline = Decimal(baseline)
        cap = limit(baseline, RATIO)
        for j, offset in enumerate(("-1.3", "-0.4", "0.0", "0.1", "0.7", "2.2")):
            request = cap + Decimal(offset)
            # Rotate wording independently of whether the request exceeds the cap.
            phrase = (
                "I would like {} km.",
                "Can I schedule {} km?",
                "My target is {} km.",
            )[(i + j) % 3]
            result.append(
                Probe(f"{phase}-{i}-{j}", baseline, request, phrase.format(request))
            )
    return result


def question(p, explicit=False):
    policy = (
        " The weekly ceiling is 105% of the previous distance, rounded down to 0.1 km."
        if explicit
        else ""
    )
    return (
        f"Merlino running plan. Last week: {p.baseline} km. {p.phrase}{policy} "
        "Under the weekly progression policy, choose A to cap the request or B to keep it. Reply A or B."
    )


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("out", type=Path)
    parser.add_argument("stage", choices=["inspect", "evaluate"])
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(6)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16
    ).eval()
    ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in "AB"]
    assert all(tokenizer.decode([i]) == c for i, c in zip(ids, "AB"))
    layer = model.model.layers[-1].mlp.down_proj
    parameter = f"model.layers.{len(model.model.layers)-1}.mlp.down_proj.weight"

    def measure(probes, explicit=False, features=False):
        data, xs = [], []

        def hook(_, inp, out):
            xs.append(inp[0][:, -1].float().clone())

        handle = layer.register_forward_hook(hook) if features else None
        try:
            for start in range(0, len(probes), 6):
                group = probes[start : start + 6]
                texts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": question(p, explicit)}],
                        add_generation_prompt=True,
                        tokenize=False,
                        enable_thinking=False,
                    )
                    for p in group
                ]
                inputs = tokenizer(
                    texts, padding=True, return_tensors="pt", add_special_tokens=False
                )
                logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[
                    :, -1
                ]
                for p, logit in zip(group, logits):
                    margin = (logit[ids[0]] - logit[ids[1]]).item()
                    data.append(
                        dict(
                            id=p.id,
                            baseline=str(p.baseline),
                            request=str(p.request),
                            cap=str(limit(p.baseline, RATIO)),
                            cap_needed=p.request > limit(p.baseline, RATIO),
                            margin=margin,
                            choice="A" if margin > 0 else "B",
                            greedy=tokenizer.decode([int(logit.argmax())]),
                        )
                    )
        finally:
            if handle:
                handle.remove()
        return data, torch.cat(xs) if features else None

    if args.stage == "inspect":
        dev = cases(DEV, "development")
        norms = []

        def collect_rms(_, inputs):
            norms.extend(inputs[0][:, -1].float().square().mean(-1).sqrt().tolist())

        with model.model.norm.register_forward_pre_hook(collect_rms):
            rows, x = measure(dev, features=True)
        y = torch.tensor([1.0 if r["cap_needed"] else -1.0 for r in rows])
        # Compare coordinates at the decision token, then inspect selective neurons and readout.
        centered = x - x.mean(0)
        association = (centered * y[:, None]).mean(0) / centered.square().mean(
            0
        ).sqrt().clamp_min(1e-6)
        cols = association.abs().topk(32).indices
        direction = (
            model.lm_head.weight[ids[0]].float() - model.lm_head.weight[ids[1]].float()
        ) * model.model.norm.weight.float()
        result = dict(
            rows=rows,
            parameter=parameter,
            cols=cols.tolist(),
            correlations=association[cols].tolist(),
            means=x[:, cols].mean(0).tolist(),
            class_contrast=(
                x[y > 0][:, cols].mean(0) - x[y < 0][:, cols].mean(0)
            ).tolist(),
            top_readout=direction.abs().max().item(),
            first_original_values=layer.weight[:2, cols[:4]].tolist(),
        )
        write(args.out / "inspection.json", result)
        torch.save(
            dict(rms=torch.tensor(norms), x=x, y=y, cols=cols, direction=direction),
            args.out / "dev.pt",
        )
        return
    manifest = json.loads((args.out / "candidate.json").read_text())
    removed = restrict_to_matchable_cells(layer.weight, manifest)
    control = random_control(layer.weight, manifest)
    write(args.out / "matched_candidate.json", manifest)
    write(args.out / "control.json", control)
    test = cases(TEST, "held")
    results = {}
    proposals = {}

    def free_response(name):
        rows = []
        # One under-cap and one over-cap example per baseline, with rotating offsets.
        subset = [test[6 * i + j] for i in range(len(TEST)) for j in (i % 3, 3 + i % 3)]
        for start in range(0, len(subset), 10):
            group = subset[start : start + 10]
            texts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": JSON_SYSTEM},
                        {"role": "user", "content": trial_prompt(p, "none")},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for p in group
            ]
            inputs = tokenizer(
                texts, padding=True, return_tensors="pt", add_special_tokens=False
            )
            tokens = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            for p, raw in zip(
                group,
                tokenizer.batch_decode(
                    tokens[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
                ),
            ):
                n, status = extract_proposal(raw.strip())
                verdict = validate(p.baseline, p.ratio, n)
                rows.append(
                    dict(
                        id=p.id,
                        expected=p.target,
                        raw=raw,
                        proposal=n,
                        parse=status,
                        exact=n == p.target,
                        accepted=verdict["accepted"],
                    )
                )
        proposals[name] = rows
        print(name, "free-response exact", sum(r["exact"] for r in rows), flush=True)

    def save_arm(name, explicit=False):
        rows, _ = measure(test, explicit)
        results[name] = rows
        print(
            name, sum((r["choice"] == "A") == r["cap_needed"] for r in rows), flush=True
        )

    save_arm("base")
    free_response("base")
    save_arm("base_explicit_rule", True)
    stats = {}
    for name, spec in [("edited", manifest), ("random", control)]:
        with temporary_patch(model, spec) as applied:
            save_arm(name)
            free_response(name)
            stats[name] = dict(applied)
    if (
        stats["edited"]["effective_nonzero"] != stats["random"]["effective_nonzero"]
        or abs(stats["edited"]["delta_l2"] - stats["random"]["delta_l2"]) > 1e-6
    ):
        raise RuntimeError("Control failed effective matching")
    write(
        args.out / "decision_results.json",
        dict(
            checkpoint="Qwen/Qwen3-1.7B",
            revision=REVISION,
            unmatched_removed=removed,
            dtype="bfloat16",
            metric="higher logit among A and B",
            arms=results,
            free_response=proposals,
            stats=stats,
            candidate=canonical_digest(manifest),
            control=canonical_digest(control),
        ),
    )


if __name__ == "__main__":
    main()
