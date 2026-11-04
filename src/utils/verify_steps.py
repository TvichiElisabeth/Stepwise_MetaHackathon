"""Check the motion-based step split.

1. Visual: saves one image per video with a row per detected step
   (first / middle / last frame), so you can see what each step contains.
2. Numbers: if data/mock/ground_truth_<name>.json exists, compares the
   detected steps with your hand-written ones.
"""
import cv2, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
import paths
from motion import load_frames, motion_scores, split_steps

def frame_at(frames, t):
    return min(frames, key=lambda f: abs(f[0] - t))[1]

def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img

def contact_sheet(name, frames, steps, out_path):
    rows = []
    for i, (s, e) in enumerate(steps, 1):
        tiles = []
        for tag, t in (("start", s), ("middle", (s + e) / 2), ("end", e)):
            img = cv2.resize(cv2.imread(frame_at(frames, t)), (320, 160))
            tiles.append(label(img, f"Step {i} {tag} {t:.1f}s"))
        rows.append(np.hstack(tiles))
    cv2.imwrite(out_path, np.vstack(rows))
    print(f"Saved {out_path}")

def compare(detected, truth):
    print(f"\nDetected {len(detected)} steps, ground truth has {len(truth)}.")
    print(f"{'Ground truth step':<32}{'best match':<14}{'overlap':>8}")
    used = set()
    for g in truth:
        gs, ge = g["start"], g["end"]
        best, best_iou = None, 0.0
        for j, (s, e) in enumerate(detected):
            inter = max(0, min(ge, e) - max(gs, s))
            union = max(ge, e) - min(gs, s)
            iou = inter / union if union else 0
            if iou > best_iou:
                best, best_iou = j, iou
        status = "ok" if best_iou >= 0.5 else ("partial" if best_iou > 0 else "MISSED")
        match = f"step {best + 1}" if best is not None else "-"
        print(f"{g['name'][:30]:<32}{match:<14}{best_iou:>7.0%}  {status}")
        if best is not None:
            used.add(best)
    merged = [j + 1 for j in range(len(detected))
              if sum(1 for g in truth
                     if max(0, min(g["end"], detected[j][1]) - max(g["start"], detected[j][0])) > 0.5 * (g["end"] - g["start"])) > 1]
    if merged:
        print(f"\nDetected step(s) {merged} contain more than one real step (merged).")

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "mouse"
    frames = load_frames(name)
    times = [t for t, _ in frames]
    steps, _, _ = split_steps(times, motion_scores(frames))
    os.makedirs(paths.plots_dir(name), exist_ok=True)
    contact_sheet(name, frames, steps, f"{paths.plots_dir(name)}/steps.jpg")

    gt_path = paths.ground_truth(name)
    if os.path.exists(gt_path):
        compare(steps, json.load(open(gt_path)))
    else:
        print(f"(No {gt_path} yet: write it to get a score.)")
