"""Create mode: narrated video -> finished SOP.

  1. frames      (frames.py)
  2. transcript  Muse Voice Transcribe
  3. steps       speech + motion merge (steps.py)
  4. keyframes   one clear image per step (keyframes.py)
  5. SOP text    Muse Spark reads narration + keyframes, writes each step
                 and says which objects matter in each photo
  6. highlight   SAM 3.1 finds those objects; boxes are drawn on the images
  7. export      HTML document (export.py)

Usage:  python src\\sop.py mouse
"""
import glob, json, os, sys
import cv2
import paths
import meta_client as mc
from frames import extract_frames
from steps import build_steps
from keyframes import add_keyframes
from export import export_html

COLORS = [(0, 170, 255), (80, 200, 60), (230, 90, 200), (255, 160, 0)]

def annotate(image_path, objects, out_path):
    """Runs SAM for each object name and draws labelled boxes."""
    img = cv2.imread(image_path)
    found = []
    for i, obj in enumerate(objects[:3]):
        boxes = sorted(mc.segment(image_path, obj), key=lambda b: -b["score"])[:1]
        for b in boxes:
            sx, sy = img.shape[1] / b.get("w", img.shape[1]), img.shape[0] / b.get("h", img.shape[0])
            p1 = (int(b["x1"] * sx), int(b["y1"] * sy))
            p2 = (int(b["x2"] * sx), int(b["y2"] * sy))
            c = COLORS[i % len(COLORS)]
            cv2.rectangle(img, p1, p2, c, 3)
            (tw, th), _ = cv2.getTextSize(obj, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            lw, lh = tw + 10, th + 10
            lx = max(0, min(p1[0], img.shape[1] - lw))          # keep it inside the right edge
            ly = p1[1] - lh if p1[1] >= lh else p1[1]           # above the box, or inside it if no room
            cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh), c, -1)
            cv2.putText(img, obj, (lx + 5, ly + lh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            found.append({"object": obj, "score": round(b["score"], 2)})
    cv2.imwrite(out_path, img)
    return found

def create_sop(name, log=print):
    video = paths.video(name)
    if not os.path.isdir(paths.frames_dir(name)):
        log("1/7 Extracting frames...")
        extract_frames(video, paths.frames_dir(name))

    log(f"2/7 Transcribing narration ({mc.BACKEND})...")
    sentences = mc.transcribe(video, name)
    os.makedirs(paths.work(name), exist_ok=True)
    json.dump(sentences, open(paths.transcript(name), "w", encoding="utf-8"), indent=2)

    log(f"3/7 Grouping narration with {mc.SPARK_MODEL}, then merging with motion...")
    groups = mc.group_steps(mc.clean_turns(sentences))
    steps = build_steps(name, groups)

    log("4/7 Picking keyframes...")
    steps = add_keyframes(name, steps)
    json.dump(steps, open(paths.steps_json(name), "w", encoding="utf-8"), indent=2)

    log(f"5/7 Writing SOP with {mc.SPARK_MODEL} ({mc.BACKEND})...")
    sop = mc.write_sop(steps)

    log("6/7 Highlighting objects with SAM 3.1...")
    images_dir = paths.output_images(name)
    os.makedirs(images_dir, exist_ok=True)
    for old in glob.glob(f"{images_dir}/step_*.jpg"):   # a previous run may have had more steps
        os.remove(old)
    by_num = {s["number"]: s for s in steps}
    for st in sop["steps"]:
        src = by_num.get(st["number"])
        if not src:
            continue
        st["start"], st["end"] = src["start"], src["end"]
        st["narration"] = src.get("text")
        img_out = f"{images_dir}/step_{st['number']:02d}.jpg"
        st["found_objects"] = annotate(src["keyframe"], st.get("visible_objects", []), img_out)
        st["image"] = img_out

    json.dump(sop, open(paths.sop_json(name), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    log("7/7 Exporting HTML...")
    html = export_html(sop, paths.sop_html(name))
    log(f"Done: {paths.sop_json(name)} and {html}")
    return sop

if __name__ == "__main__":
    create_sop(sys.argv[1] if len(sys.argv) > 1 else "mouse")
