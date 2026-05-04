"""Read a speedscope JSON and print top-N by self-time + categorical aggregation.

Used by recap.md generation per CLAUDE.md Section 9. Single-profile mode by default;
pass --compare <other.json> for two-profile mode.

Self-time = weight of samples whose leaf frame is this function.
Total time = weight of samples whose stack contains this function (counted once per stack).
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


# Categorical aggregation per Section 9 Step 3. Order matters: first match wins.
CATEGORIES = [
    ("compute_torch", [
        r"torch[/\\.]",
        r"_C\._",
        r"aten::",
        r"backward",
        r"autograd",
    ]),
    ("compression", [
        r"\bzlib\b", r"deflate", r"gzip\b",
    ]),
    ("serialization", [
        r"\bjson\b.*encoder", r"\bjson\b.*decoder", r"\bjson\b/__init__",
        r"json_pack", r"json_unpack",
        r"_serialize", r"\bpickle\b",
        r"numpy.*serialize", r"\.tobytes\b",
    ]),
    ("communication_websockets", [
        r"websockets[/\\.]", r"send_websocket", r"recv_websocket",
    ]),
    ("declearn_internal", [
        r"declearn[/\\.]optimizer", r"declearn[/\\.]model",
        r"declearn[/\\.]aggregator", r"declearn[/\\.]messaging",
        r"declearn[/\\.]main", r"declearn[/\\.]secagg",
        r"declearn[/\\.]fairness", r"declearn[/\\.]metrics",
        r"declearn[/\\.]dataset", r"declearn[/\\.]quickrun",
        r"declearn[/\\.]communication", r"declearn[/\\.]utils",
        r"Vector\b", r"NumpyVector", r"TorchVector",
    ]),
    ("asyncio_eventloop", [
        r"asyncio[/\\.]", r"selectors[/\\.]", r"_run_once",
    ]),
    ("imports_startup", [
        r"<module>", r"importlib", r"_bootstrap", r"<frozen importlib",
        r"_find_and_load", r"_load_unlocked",
    ]),
]


def categorize(frame_str):
    for cat, patterns in CATEGORIES:
        for pat in patterns:
            if re.search(pat, frame_str):
                return cat
    return "other"


def load(path):
    with open(path) as f:
        d = json.load(f)
    frames = d["shared"]["frames"]
    profiles = d["profiles"]
    return d, frames, profiles


def frame_label(f):
    name = f.get("name", "?")
    file = f.get("file", "")
    if file:
        return f"{name}  ({file})"
    return name


def collect(profiles, frames):
    """Reconstruct stacks from samples by walking the speedscope events.

    speedscope `evented` profiles use samples = list of stacks (each is a list of frame ids).
    Some speedscope variants use 'sampled' type — handle both.
    """
    self_time = defaultdict(float)
    total_time = defaultdict(float)
    total_weight = 0.0
    stack_count = 0

    for prof in profiles:
        samples = prof.get("samples", [])
        weights = prof.get("weights", [])
        if not samples:
            continue
        for stack, w in zip(samples, weights):
            total_weight += w
            stack_count += 1
            if not stack:
                continue
            # Self time: leaf frame
            self_time[stack[-1]] += w
            # Total time: every frame in stack, deduped
            for fid in set(stack):
                total_time[fid] += w
    return self_time, total_time, total_weight, stack_count


def fmt_pct(num, denom):
    if denom <= 0:
        return "0.00%"
    return f"{100.0 * num / denom:5.2f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("speedscope_path")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--compare", default=None,
                   help="Second speedscope JSON for A/B comparison")
    args = p.parse_args()

    d, frames, profiles = load(args.speedscope_path)
    self_time, total_time, total_w, n_stacks = collect(profiles, frames)
    unit = profiles[0].get("unit", "samples") if profiles else "samples"
    sample_to_sec = 1e-9 if unit == "nanoseconds" else (
        1e-6 if unit == "microseconds" else (
            1e-3 if unit == "milliseconds" else 1.0
        )
    )
    print(f"\n=== {Path(args.speedscope_path).name} ===")
    print(f"unit:         {unit}")
    print(f"total weight: {total_w:.0f} {unit} = {total_w * sample_to_sec:.2f} s")
    print(f"stacks:       {n_stacks}")
    print(f"unique frames in shared: {len(frames)}")

    # Top-N by self
    print(f"\n--- Top-{args.top} by self-time ---")
    print(f"{'idx':>4}  {'self_s':>8}  {'self_%':>7}  {'total_s':>8}  {'total_%':>8}  frame")
    ranked = sorted(self_time.items(), key=lambda kv: kv[1], reverse=True)
    for i, (fid, w) in enumerate(ranked[:args.top]):
        f = frames[fid]
        ts = total_time[fid]
        print(f"{i+1:>4}  {w*sample_to_sec:>8.3f}  {fmt_pct(w, total_w):>7}  "
              f"{ts*sample_to_sec:>8.3f}  {fmt_pct(ts, total_w):>8}  {frame_label(f)}")

    # Top-N by total
    print(f"\n--- Top-{args.top} by total-time ---")
    print(f"{'idx':>4}  {'total_s':>8}  {'total_%':>8}  frame")
    rt = sorted(total_time.items(), key=lambda kv: kv[1], reverse=True)
    for i, (fid, w) in enumerate(rt[:args.top]):
        f = frames[fid]
        print(f"{i+1:>4}  {w*sample_to_sec:>8.3f}  {fmt_pct(w, total_w):>8}  {frame_label(f)}")

    # Categorical (by self-time)
    print(f"\n--- Categorical aggregation (by self-time) ---")
    cat_self = defaultdict(float)
    for fid, w in self_time.items():
        cat_self[categorize(frame_label(frames[fid]))] += w
    print(f"{'category':<25}  {'self_s':>8}  {'self_%':>7}")
    for cat, w in sorted(cat_self.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{cat:<25}  {w*sample_to_sec:>8.3f}  {fmt_pct(w, total_w):>7}")

    if args.compare:
        d2, frames2, profiles2 = load(args.compare)
        st2, tt2, tw2, _ = collect(profiles2, frames2)
        unit2 = profiles2[0].get("unit", "samples") if profiles2 else "samples"
        s2s_2 = 1e-9 if unit2 == "nanoseconds" else (
            1e-6 if unit2 == "microseconds" else (
                1e-3 if unit2 == "milliseconds" else 1.0
            )
        )
        # Build name → time maps for comparison (frame indices differ across files)
        def name_map(times, frames):
            m = defaultdict(float)
            for fid, w in times.items():
                m[frame_label(frames[fid])] += w
            return m
        n1 = name_map(self_time, frames)
        n2 = name_map(st2, frames2)
        all_names = set(n1) | set(n2)
        rows = []
        for name in all_names:
            a = n1.get(name, 0.0) * sample_to_sec
            b = n2.get(name, 0.0) * s2s_2
            rows.append((name, a, b, b - a))
        rows.sort(key=lambda r: abs(r[3]), reverse=True)
        print(f"\n=== A/B (B − A) by self-time, top 25 absolute deltas ===")
        print(f"baseline total: {total_w*sample_to_sec:.2f}s   "
              f"variant total: {tw2*s2s_2:.2f}s   "
              f"delta: {tw2*s2s_2 - total_w*sample_to_sec:+.2f}s")
        print(f"{'baseline_s':>10}  {'variant_s':>10}  {'delta_s':>10}  frame")
        for name, a, b, delta in rows[:25]:
            print(f"{a:>10.3f}  {b:>10.3f}  {delta:>+10.3f}  {name[:120]}")


if __name__ == "__main__":
    main()
