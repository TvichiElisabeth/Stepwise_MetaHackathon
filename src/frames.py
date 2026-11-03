import cv2, os, sys

CROP_BOTTOM = 0.84   # keep the top 84% of the image (drops the laptop edge)

def crop(frame, keep=CROP_BOTTOM):
    h = frame.shape[0]
    return frame[:int(h * keep), :]

def preview(video_path, at_seconds=19, out_path="data/preview.jpg"):
    """Save one frame (with the crop line drawn) so you can check the crop."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Can't open {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, at_seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Couldn't read a frame at that time")
    y = int(frame.shape[0] * CROP_BOTTOM)
    shown = frame.copy()
    cv2.line(shown, (0, y), (shown.shape[1], y), (0, 0, 255), 4)   # red = crop line
    both = cv2.vconcat([
        cv2.resize(shown, (960, 540)),
        cv2.resize(crop(frame), (960, int(960 * y / frame.shape[1]))),
    ])
    cv2.imwrite(out_path, both)
    print(f"Saved {out_path}: top = original with red crop line, bottom = cropped result")

def extract_frames(video_path, out_dir, fps_sample=2, max_frames=None):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Can't open {video_path} (try converting to H.264 mp4)")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(video_fps / fps_sample))
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            t = i / video_fps
            frame = crop(frame)
            frame = cv2.resize(frame, (960, int(960 * frame.shape[0] / frame.shape[1])))
            path = os.path.join(out_dir, f"frame_{t:07.2f}.jpg")
            cv2.imwrite(path, frame)
            frames.append((t, path))
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames

if __name__ == "__main__":
    # usage:
    #   python src\frames.py mouse preview   -> one frame to check the crop
    #   python src\frames.py mouse test      -> only the first 10 frames
    #   python src\frames.py mouse           -> all frames
    name = sys.argv[1] if len(sys.argv) > 1 else "mouse"
    mode = sys.argv[2] if len(sys.argv) > 2 else "all"
    video = f"data/videos/{name}.mp4"

    if mode == "preview":
        preview(video)
    else:
        limit = 10 if mode == "test" else None
        f = extract_frames(video, f"data/frames/{name}", max_frames=limit)
        print(f"Extracted {len(f)} frames from {name}")