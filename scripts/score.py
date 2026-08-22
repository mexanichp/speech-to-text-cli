#!/usr/bin/env python3
"""Score a transcript against a reference, on words and on sentence boundaries.

Two numbers, because one of them cannot see the defect this project exists to
repair. Stripping punctuation to compare word streams makes a sentence split in
two identical to one left whole, and splitting is exactly what a short audio
buffer does to a transcript. So boundaries are scored separately.

``WER``
    Levenshtein over case-folded, punctuation-stripped words.

``boundary F1``
    A boundary is the gap after the last word of a sentence. Hypothesis
    boundaries are mapped onto reference word positions through the same
    alignment the WER uses, so a boundary counts as found only if it sits in
    the same place in the speech, not merely at the same word count.

Usage::

    .venv/bin/python scripts/score.py --ref scratch/oracle_2.md hyp.txt
    .venv/bin/python scripts/score.py --ref a.md b.txt --wake Luna --diff
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata

TERMINALS = ".?!。！？"


def normalize(word: str) -> str:
    """Case-fold and strip everything that is not a letter, digit or apostrophe."""
    word = unicodedata.normalize("NFKC", word).casefold()
    return re.sub(r"[^\w']", "", word, flags=re.UNICODE)


def parse(text: str, wake: str | None) -> tuple[list[str], set[int]]:
    """Return normalized words and the set of word indices a sentence ends at.

    ``wake`` drops sentences naming the assistant. A spoken command is an
    instruction, not dictation, so the live pipeline correctly leaves it out of
    the document and a reference transcribed straight from the audio correctly
    leaves it in. Comparing the two without this scores the feature as an error.
    """
    sentences = [s for s in re.split(rf"(?<=[{re.escape(TERMINALS)}])\s+", text.strip()) if s]
    named = normalize(wake) if wake else None

    words: list[str] = []
    ends: set[int] = set()
    for sentence in sentences:
        got = [w for w in (normalize(t) for t in sentence.split()) if w]
        if not got or (named and named in got):
            continue
        words.extend(got)
        ends.add(len(words))
    # The end of the passage is a boundary nobody can get wrong, so it says
    # nothing about either transcript.
    ends.discard(len(words))
    return words, ends


def align(ref: list[str], hyp: list[str]) -> tuple[int, int, int, list[tuple[int | None, int | None]]]:
    """Levenshtein-align. Returns (substitutions, deletions, insertions, pairs)."""
    n, m = len(ref), len(hyp)
    # cost[i][j] = edits turning ref[:i] into hyp[:j]; back[i][j] = how.
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0], back[i][0] = i, "d"
    for j in range(1, m + 1):
        cost[0][j], back[0][j] = j, "i"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost[i][j], back[i][j] = cost[i - 1][j - 1], "="
                continue
            options = (
                (cost[i - 1][j - 1] + 1, "s"),
                (cost[i - 1][j] + 1, "d"),
                (cost[i][j - 1] + 1, "i"),
            )
            cost[i][j], back[i][j] = min(options)

    subs = dels = ins = 0
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i or j:
        op = back[i][j]
        if op == "=":
            i, j = i - 1, j - 1
            pairs.append((i, j))
        elif op == "s":
            i, j = i - 1, j - 1
            pairs.append((i, j))
            subs += 1
        elif op == "d":
            i -= 1
            pairs.append((i, None))
            dels += 1
        else:
            j -= 1
            pairs.append((None, j))
            ins += 1
    pairs.reverse()
    return subs, dels, ins, pairs


def boundary_scores(
    ref: list[str], hyp: list[str], ref_ends: set[int], hyp_ends: set[int],
    pairs: list[tuple[int | None, int | None]],
) -> tuple[float, float, float]:
    """Map hypothesis boundaries onto reference positions and score them.

    A hypothesis boundary after hypothesis word ``j`` lands after whichever
    reference word ``j`` aligned to. An inserted word carries no reference
    position, so the boundary attaches to the last reference word before it.
    """
    # hyp word index -> reference index it follows.
    at: dict[int, int] = {}
    last = -1
    for r, h in pairs:
        if r is not None:
            last = r
        if h is not None:
            at[h] = last

    mapped = {at.get(j - 1, -1) + 1 for j in hyp_ends}
    mapped.discard(0)

    hit = len(mapped & ref_ends)
    precision = hit / len(mapped) if mapped else float(len(ref_ends) == 0)
    recall = hit / len(ref_ends) if ref_ends else float(len(mapped) == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def show_diff(ref: list[str], hyp: list[str], pairs) -> None:
    """Print the alignment, one flagged line per run of differences."""
    line: list[str] = []
    for r, h in pairs:
        if r is not None and h is not None and ref[r] == hyp[h]:
            if line:
                print("    " + " ".join(line))
                line = []
            continue
        if r is not None and h is not None:
            line.append(f"[{ref[r]}->{hyp[h]}]")
        elif r is not None:
            line.append(f"[-{ref[r]}]")
        else:
            line.append(f"[+{hyp[h]}]")
    if line:
        print("    " + " ".join(line))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("hyp", help="transcript under test")
    ap.add_argument("--ref", required=True, help="reference transcript")
    ap.add_argument("--wake", default=None, help="drop sentences naming this assistant")
    ap.add_argument("--diff", action="store_true", help="print the word alignment")
    args = ap.parse_args()

    ref_text = open(args.ref, encoding="utf-8").read()
    hyp_text = open(args.hyp, encoding="utf-8").read()

    ref, ref_ends = parse(ref_text, args.wake)
    hyp, hyp_ends = parse(hyp_text, args.wake)
    if not ref:
        sys.exit(f"{args.ref}: no words")

    subs, dels, ins, pairs = align(ref, hyp)
    wer = (subs + dels + ins) / len(ref)
    precision, recall, f1 = boundary_scores(ref, hyp, ref_ends, hyp_ends, pairs)

    print(f"ref {args.ref}   {len(ref)} words, {len(ref_ends)} boundaries")
    print(f"hyp {args.hyp}   {len(hyp)} words, {len(hyp_ends)} boundaries")
    print(f"WER          {wer:6.1%}   ({subs} sub, {dels} del, {ins} ins)")
    print(f"boundary F1  {f1:6.2f}     (P {precision:.2f}, R {recall:.2f})")

    if args.diff:
        print("\ndifferences:")
        show_diff(ref, hyp, pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
