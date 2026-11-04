import cv2, glob, os, sys
import numpy as np
import matplotlib.pyplot as plt
import paths

def load_frames(name):
    files = sorted(glob.glob(f"{paths.frames_dir(name)}/frame_*.jpg"))
    times = [float(os.path.basename(p)[6:-4]) for p in files]
    return list(zip(times, files))

def motion_scores(frames):
    scores, prev = [], None
    for _, p in frames:
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY), (21, 21), 0)
        scores.append(0.0 if prev is None else float(np.mean(cv2.absdiff(g, prev))))
        prev = g
    return np.array(scores)

def split_steps(times, scores, still_ratio=0.35, min_pause_s=1.0, min_step_s=2.0):
    """Active stretches separated by pauses become steps."""
    smooth = np.convolve(scores, np.ones(3) / 3, mode="same")
    threshold = still_ratio * np.percentile(smooth, 90)
    active = smooth > threshold

    steps, start, still_since = [], None, None
    for t, a in zip(times, active):
        if a:
            if start is None:
                start = t
            still_since = None
        elif start is not None:
            still_since = still_since if still_since is not None else t
            if t - still_since >= min_pause_s:
                if still_since - start >= min_step_s:
                    steps.append((start, still_since))
                start, still_since = None, None
    if start is not None and times[-1] - start >= min_step_s:
        steps.append((start, times[-1]))
    return steps, threshold, smooth

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "mouse"
    frames = load_frames(name)
    times = [t for t, _ in frames]
    scores = motion_scores(frames)
    steps, thr, smooth = split_steps(times, scores)

    print(f"Found {len(steps)} steps:")
    for i, (s, e) in enumerate(steps, 1):
        print(f"  Step {i}: {s:6.1f}s -> {e:6.1f}s")

    plt.figure(figsize=(12, 4))
    plt.plot(times, smooth, label="motion")
    plt.axhline(thr, color="red", linestyle="--", label="threshold")
    for s, e in steps:
        plt.axvspan(s, e, alpha=0.2, color="green")
    plt.xlabel("time (s)"); plt.ylabel("motion"); plt.legend(); plt.tight_layout()
    os.makedirs(paths.plots_dir(name), exist_ok=True)
    plt.savefig(f"{paths.plots_dir(name)}/motion.png")
    plt.show()