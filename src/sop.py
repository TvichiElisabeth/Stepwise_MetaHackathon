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
import json, os, sys
import cv2
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
            cv2.rectangle(img, (p1[0], max(0, p1[1] - th - 10)), (p1[0] + tw + 10, p1[1]), c, -1)
            cv2.putText(img, obj, (p1[0] + 5, p1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            found.append({"object": obj, "score": round(b["score"], 2)})
    cv2.imwrite(out_path, img)
    return found

def create_sop(name, log=print):
    video = f"data/videos/{name}.mp4"
    if not os.path.isdir(f"data/frames/{name}"):
        log("1/7 Extracting frames...")
        extract_frames(video, f"data/frames/{name}")

    log(f"2/7 Transcribing narration ({mc.BACKEND})...")
    sentences = mc.transcribe(video, name)
    os.makedirs("data/transcripts", exist_ok=True)
    json.dump(sentences, open(f"data/transcripts/{name}.json", "w", encoding="utf-8"), indent=2)

    log("3/7 Merging speech and motion into steps...")
    steps = build_steps(name, sentences)

    log("4/7 Picking keyframes...")
    steps = add_keyframes(name, steps)

    log(f"5/7 Writing SOP with {mc.SPARK_MODEL} ({mc.BACKEND})...")
    sop = mc.write_sop(steps)

    log("6/7 Highlighting objects with SAM 3.1...")
    out_dir = f"data/sop/{name}"
    os.makedirs(out_dir, exist_ok=True)
    by_num = {s["number"]: s for s in steps}
    for st in sop["steps"]:
        src = by_num.get(st["number"])
        if not src:
            continue
        st["start"], st["end"] = src["start"], src["end"]
        st["narration"] = src.get("text")
        img_out = f"{out_dir}/step_{st['number']:02d}.jpg"
        st["found_objects"] = annotate(src["keyframe"], st.get("visible_objects", []), img_out)
        st["image"] = img_out

    json.dump(sop, open(f"{out_dir}/sop.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    log("7/7 Exporting HTML...")
    html = export_html(sop, f"{out_dir}/sop.html")
    log(f"Done: {out_dir}/sop.json and {html}")
    return sop

if __name__ == "__main__":
    create_sop(sys.argv[1] if len(sys.argv) > 1 else "mouse")
