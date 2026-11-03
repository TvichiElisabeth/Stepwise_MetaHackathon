"""Merge narration (what each step is) with motion (when the work happens).

Rules:
1. Each sentence is assigned to a motion burst:
   - the burst it falls inside, or
   - the next burst starting within LOOKAHEAD_S seconds (people talk, then act), or
   - the nearest burst.
2. A burst with N sentences is split into N steps at the N-1 calmest
   moments inside it (local minima of motion), because people pause
   briefly between sub-steps even when they don't fully stop.
3. Step start = earlier of (sentence start, burst start); end = burst end.
4. Bursts with no sentence become "unnarrated" steps; sentences with no
   burst become short steps, so nothing is silently dropped.
"""
import json, os, sys
import numpy as np
from motion import load_frames, motion_scores, split_steps

LOOKAHEAD_S = 4.0

def assign(sentences, bursts):
    groups = {i: [] for i in range(len(bursts))}
    orphans = []
    for s in sentences:
        t = s["start"]
        inside = [i for i, (a, b) in enumerate(bursts) if a <= t <= b]
        ahead = [i for i, (a, _) in enumerate(bursts) if 0 <= a - t <= LOOKAHEAD_S]
        if inside:
            groups[inside[0]].append(s)
        elif ahead:
            groups[ahead[0]].append(s)
        elif bursts:
            dist = [min(abs(t - a), abs(t - b)) for a, b in bursts]
            i = int(np.argmin(dist))
            (groups[i] if dist[i] <= 2 * LOOKAHEAD_S else orphans).append(s)
        else:
            orphans.append(s)
    return groups, orphans

def calm_points(times, smooth, start, end, n):
    """Times of the n calmest local minima inside (start, end), in order."""
    idx = [i for i, t in enumerate(times) if start + 1.0 < t < end - 1.0]
    minima = [i for i in idx if smooth[i] <= smooth[i - 1] and smooth[i] <= smooth[i + 1]]
    minima = sorted(minima, key=lambda i: smooth[i])[:n]
    return sorted(times[i] for i in minima)

def build_steps(name):
    frames = load_frames(name)
    times = [t for t, _ in frames]
    bursts, _, smooth = split_steps(times, motion_scores(frames))
    sentences = json.load(open(f"data/mock/transcript_{name}.json"))
    groups, orphans = assign(sentences, bursts)

    steps = []
    for i, (a, b) in enumerate(bursts):
        group = sorted(groups[i], key=lambda s: s["start"])
        if not group:
            steps.append({"start": a, "end": b, "text": None, "source": "motion only"})
            continue
        cuts = calm_points(times, smooth, a, b, len(group) - 1)
        if len(cuts) < len(group) - 1:          # not enough calm points: use speech times
            cuts = [s["start"] for s in group[1:]]
        edges = [min(a, group[0]["start"])] + cuts + [b]
        for k, s in enumerate(group):
            src = "speech + motion" if len(group) == 1 else "speech + motion (split)"
            steps.append({"start": edges[k], "end": edges[k + 1], "text": s["text"], "source": src})
    for s in orphans:
        steps.append({"start": s["start"], "end": s["start"] + 2.0, "text": s["text"], "source": "speech only"})

    steps.sort(key=lambda s: s["start"])
    for n, s in enumerate(steps, 1):
        s["number"] = n
    return steps

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "mouse"
    steps = build_steps(name)
    print(f"{len(steps)} steps:")
    for s in steps:
        print(f"  {s['number']}. {s['start']:5.1f}s -> {s['end']:5.1f}s  [{s['source']}]  {s['text']}")
    os.makedirs("data/steps", exist_ok=True)
    out = f"data/steps/{name}_steps.json"
    json.dump(steps, open(out, "w"), indent=2)
    print(f"Saved {out}")
