"""Pick one clear image per step for the SOP.


To choose the best photo for each step, the script scores candidate frames on three things: 
how sharp the image is, how still the hands are, and how close the frame is to the middle of the step. 

   -Sharpness uses the variance of the Laplacian, a standard computer-vision blur measure (strong, varied edges mean a sharp image). 
   -That value is divided by the amount of motion since the previous frame, because a frame taken mid-movement is usually blurred and shows an in-between moment.
   -The result is then multiplied by a centrality factor that is 1.0 at the middle of the step and 0.5 at its edges. The centrality factor and the restriction to the middle half of each step were added after testing on the mouse video.
     Without them, the formula kept choosing frames from just before the action began, which are sharp and still precisely because nothing has happened yet. 

The weights are a heuristic tuned by inspecting the results, not optimized values.
"""
import cv2, glob, json, os, shutil, sys
import numpy as np
import paths
from motion import load_frames

def sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()      # higher = sharper

def pick_keyframe(frames, start, end):
    """Sharp, fairly still frame close to the middle of the step."""

    #Setting up the time window
    span = end - start                                # length of step
    mid = (start + end) / 2                           # middle of the step
    lo, hi = start + 0.25 * span, end - 0.25 * span   # interval of the step
    best, best_score, prev = None, -1, None

    #Iterating through the frames; t is time and p is the image file
    for i, (t, p) in enumerate(frames):
        img = cv2.imread(p) if lo - 0.6 <= t <= hi else None   # to measure motion for the first candidate, the prev frame is needed too.Starting 0.6 s early loads exactly one extra frame, since frames are 0.5 s apart.
        if img is None:
            continue
        g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (21, 21), 0)   #grayscale blurred copy only for measuring motion, the blur stops camera noise from counting as movement
        if t < lo:                    # only used as the "previous" frame
            prev = g
            continue
        #Scoring each candidate
        motion = 0 if prev is None else float(np.mean(cv2.absdiff(g, prev))) #motion is the average pixel difference from the previous frame. Near 0 means the hands are still; high means moving. The current frame then becomes prev for the next comparison.
        prev = g
        centrality = 1 - 0.5 * abs(t - mid) / (span / 2)     # 1 at the middle
        score = sharpness(img) / (1 + motion) * centrality
        if score > best_score:
            best, best_score = (t, p), score
    return best or min(frames, key=lambda f: abs(f[0] - mid))

def add_keyframes(name, steps):
    """Adds "keyframe" and "keyframe_time" to each step; saves a preview sheet."""
    frames = load_frames(name)

    out_dir = paths.keyframes_dir(name)
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(f"{out_dir}/step_*.jpg"):      # a previous run may have had more steps
        os.remove(old)
    tiles = []
    for s in steps:
        t, p = pick_keyframe(frames, s["start"], s["end"])
        dest = os.path.join(out_dir, f"step_{s['number']:02d}.jpg")
        shutil.copy(p, dest)
        s["keyframe"] = dest.replace("\\", "/")
        s["keyframe_time"] = t
        print(f"  Step {s['number']}: keyframe at {t:5.1f}s -> {dest}")
        tile = cv2.resize(cv2.imread(p), (320, 160))
        cv2.rectangle(tile, (0, 0), (320, 26), (0, 0, 0), -1)
        cv2.putText(tile, f"Step {s['number']} ({t:.1f}s)", (6, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(tile)

    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    os.makedirs(paths.plots_dir(name), exist_ok=True)
    cv2.imwrite(f"{paths.plots_dir(name)}/keyframes.jpg", sheet)
    return steps

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "mouse"
    steps_path = paths.steps_json(name)
    steps = add_keyframes(name, json.load(open(steps_path)))
    json.dump(steps, open(steps_path, "w"), indent=2)
    print(f"Updated {steps_path} and saved {paths.plots_dir(name)}/keyframes.jpg")