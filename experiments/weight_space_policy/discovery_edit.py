"""Freeze a 128-by-32 decision-readout edit from development observations."""

import argparse
import json
from pathlib import Path

import torch
from .patch import parse_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    observed = torch.load(directory / "dev.pt", weights_only=True)
    report = json.loads((directory / "inspection.json").read_text())
    x, y, rms = (observed[k] for k in ("x", "y", "rms"))
    # The final RMSNorm divides the residual by this RMS before the token readout.
    x = x / rms[:, None]
    readout = observed["direction"]
    rows = readout.abs().topk(128).indices
    readout_sum = readout[rows].abs().sum()
    # Use a strongly active, stable unit to counter the original keep bias.
    mean = x.mean(0)
    sd = x.std(0).clamp_min(1e-6)
    anchor = (mean.abs() / (sd + 0.01)).argmax()
    relative = x - x[:, anchor, None] * (mean / mean[anchor])
    contrast = (relative * y[:, None]).mean(0)
    reliability = contrast.abs() / relative.std(0).clamp_min(1e-6)
    ranking = reliability * contrast.abs()
    ranking[anchor] = -1
    features = ranking.topk(31).indices
    key = contrast[features].sign() * 0.045
    margin = torch.tensor([r["margin"] for r in report["rows"]])
    offset = -(margin.mean() / readout_sum + mean[features] @ key) / mean[anchor]
    key = torch.cat((key, offset.reshape(1)))
    key *= min(1.0, 0.05 / float(key.abs().max()))
    cols = torch.cat((features, anchor.reshape(1)))
    delta = readout[rows].sign()[:, None] * key
    cells = [
        [int(r), int(c), float(d)]
        for r, values in zip(rows, delta)
        for c, d in zip(cols, values)
        if float(d)
    ]
    # Round down the representational overshoot of 0.05 in float32.
    for cell in cells:
        cell[2] = max(-0.05, min(0.05, cell[2]))
    manifest = {
        "version": "wsp-edit-v0",
        "hypothesis": "Increase the cap-versus-keep readout along last-layer MLP features with consistent development class contrast after removing common activation.",
        "edits": [{"parameter": report["parameter"], "cells": cells}],
    }
    parse_manifest(manifest)
    (directory / "candidate.json").write_text(json.dumps(manifest) + "\n")
    predicted = margin + readout_sum * (x[:, cols] @ key)
    print(
        "Development linearized margin accuracy",
        int(((predicted > 0) == (y > 0)).sum()),
        "/",
        len(y),
    )
    print("Development residual RMS range", float(rms.min()), float(rms.max()))


if __name__ == "__main__":
    main()
