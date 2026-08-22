#!/usr/bin/env python3
"""Run the pipeline several times per configuration and report the spread.

Why this exists
---------------
CLAUDE.md §8 records that the pipeline is not deterministic: inference timing
decides when ticks and trims land, which decides what is in the buffer, which
decides the text. It also records the trap, which is that it is not reliably
*in*determinate either, so a pair of matching runs says nothing about the next
one.

Measured on recording 3, boundary precision spans 0.11 across three runs of one
binary. Every single-run comparison made before this script existed was
therefore unreadable, including two that were written up as findings.

Two things make a comparison mean something, and this script does both.

**Repeat.** A configuration is a distribution, not a number, so it is reported
as one: median, full range, and every run.

**Interleave.** Arms are run round-robin rather than one arm to completion, so
thermal state, background load and anything else that drifts over an hour lands
on both arms equally instead of on whichever ran second.

Usage::

    scripts/bench.py scratch/g3.wav --ref scratch/oracle_3.md --runs 5 \\
        --arm shipping \\
        --arm no-mark:STT_ABLATE=seam-mark

An arm is ``label`` or ``label:KEY=VALUE[,KEY=VALUE...]``, the environment
overlaid on the run. ``--wake`` is passed through to the scorer.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINARY = HERE.parent / "target" / "release" / "speech-to-text-cli"


def parse_arm(spec: str) -> tuple[str, dict[str, str]]:
    """``label`` or ``label:K=V;K=V`` into a name and an environment overlay.

    Pairs are separated by ``;`` rather than ``,`` because the variable this
    exists to set, ``STT_ABLATE``, takes a comma-separated list of its own.
    """
    label, _, rest = spec.partition(":")
    env: dict[str, str] = {}
    for pair in filter(None, (p.strip() for p in rest.split(";"))):
        key, _, value = pair.partition("=")
        env[key.strip()] = value.strip()
    return label.strip(), env


def run_once(wav: Path, env: dict[str, str], out: Path, language: str) -> None:
    """One pipeline run. Paced at realtime by `--simulate`, so it takes as long
    as the recording; that pacing is the thing under test and must not change."""
    child = dict(os.environ, **env)
    child.pop("STT_TRACE", None)
    with out.open("w") as handle:
        subprocess.run(
            [str(BINARY), "--simulate", str(wav), "--language", language],
            stdout=handle,
            stderr=subprocess.DEVNULL,
            env=child,
            check=True,
        )


def score(out: Path, ref: Path, wake: str | None) -> dict[str, float]:
    """WER and boundary precision/recall, via the scorer the project already has."""
    cmd = [sys.executable, str(HERE / "score.py"), str(out), "--ref", str(ref)]
    if wake:
        cmd += ["--wake", wake]
    text = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

    got: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("WER"):
            got["wer"] = float(line.split()[1].rstrip("%"))
        elif line.startswith("boundary F1"):
            numbers = re.findall(r"\d+\.\d+", line)
            if len(numbers) >= 3:
                got["f1"], got["p"], got["r"] = (float(n) for n in numbers[:3])
    return got


def summarize(name: str, values: list[float], lower_is_better: bool) -> str:
    if not values:
        return f"{name:>4} --"
    lo, hi = min(values), max(values)
    arrow = "lower better" if lower_is_better else "higher better"
    return f"{name:>4} median {statistics.median(values):6.2f}   range {lo:6.2f}-{hi:6.2f}   ({arrow})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wav")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--runs", type=int, default=5, help="runs per arm")
    ap.add_argument("--arm", action="append", required=True, help="label[:K=V;...]")
    ap.add_argument("--wake")
    ap.add_argument("--language", default="en")
    ap.add_argument("--keep", help="directory for the transcripts, else a temp one")
    args = ap.parse_args()

    if not BINARY.exists():
        print(f"no binary at {BINARY}; cargo build --release", file=sys.stderr)
        return 1

    arms = [parse_arm(a) for a in args.arm]
    results: dict[str, list[dict[str, float]]] = {label: [] for label, _ in arms}

    keep = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="bench-"))
    keep.mkdir(parents=True, exist_ok=True)

    # Round-robin, not arm-at-a-time. An hour of runs drifts; interleaving makes
    # the drift common to every arm instead of confounded with one of them.
    for i in range(args.runs):
        for label, env in arms:
            out = keep / f"{label}-{i + 1}.txt"
            print(f"  run {i + 1}/{args.runs}  {label} ...", end="", flush=True, file=sys.stderr)
            run_once(Path(args.wav), env, out, args.language)
            got = score(out, Path(args.ref), args.wake)
            results[label].append(got)
            print(
                f" WER {got.get('wer', float('nan')):.1f}%  P {got.get('p', float('nan')):.2f}"
                f"  R {got.get('r', float('nan')):.2f}",
                file=sys.stderr,
            )

    print(f"\n{args.wav}  vs  {args.ref}   {args.runs} run(s) per arm, interleaved\n")
    for label, _ in arms:
        runs = results[label]
        print(f"{label}")
        for metric, lower in (("wer", True), ("p", False), ("r", False)):
            print("   " + summarize(metric.upper(), [r[metric] for r in runs if metric in r], lower))
        print()

    if len(arms) > 1:
        # Pairwise, not all-at-once. Asking whether *every* arm separates from
        # every other reports one verdict for the whole row, and a single
        # overlapping pair then hides every separation beside it: a sweep where
        # one point is clearly better than the rest reads as "no result".
        print("pairwise, and only a separated pair says anything:")
        for metric in ("wer", "p", "r"):
            spans = {
                label: (min(v), max(v))
                for label, runs in results.items()
                if (v := [r[metric] for r in runs if metric in r])
            }
            print(f"   {metric.upper()}")
            for label, (lo, hi) in spans.items():
                print(f"      {label:<12} {lo:.2f}-{hi:.2f}")
            items = list(spans.items())
            for i, (a, (alo, ahi)) in enumerate(items):
                for b, (blo, bhi) in items[i + 1 :]:
                    apart = ahi < blo or bhi < alo
                    better = ""
                    if apart:
                        wins = a if (ahi < blo) == (metric == "wer") else b
                        better = f", {wins} better"
                    verdict = f"separated{better}" if apart else "overlapping"
                    print(f"      {a} vs {b}: {verdict}")

    print(f"\ntranscripts in {keep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
