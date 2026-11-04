"""All calls to Meta models live here.

BACKEND (set in .env):
  meta_api  -> real calls to Meta Model API (default)
  mock      -> no API calls; uses data/mock files and simple fakes

Every real response is cached in cache/, keyed by its inputs, so each
video/image is paid for only once. Re-runs and demos are free.
"""
import base64, hashlib, json, os, re, subprocess
from dotenv import load_dotenv

load_dotenv()
BACKEND = os.getenv("BACKEND", "meta_api")
SPARK_MODEL = os.getenv("SPARK_MODEL", "muse-spark-1.3")
SAM_MODEL = "sam-3.1"
ASR_MODEL = "muse-voice-transcribe-1.0"
BASE_URL = "https://api.meta.ai/v1"
CACHE_DIR = "cache"

# ---------- helpers ----------

def _key(*parts):
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, str) and os.path.isfile(p):
            h.update(open(p, "rb").read())
        else:
            h.update(json.dumps(p, sort_keys=True).encode())
    return h.hexdigest()[:24]

def _cached(name, key, fn):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}_{key}.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    result = fn()
    json.dump(result, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return result

def _client():
    from openai import OpenAI
    return OpenAI(base_url=BASE_URL, api_key=os.environ["MODEL_API_KEY"])

def _data_url(image_path):
    b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    return f"data:image/jpeg;base64,{b64}"

def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"

def _log_usage(model, resp):
    try:
        from usage_tracker import log_usage
        log_usage(model, resp)
    except Exception:
        pass

# ---------- 1. Muse Voice Transcribe: audio -> timestamped sentences ----------

def transcribe(video_path, name=None):
    """Returns [{"start": s, "end": s, "text": str}, ...]"""
    if BACKEND == "mock":
        return json.load(open(f"data/mock/transcript_{name}.json", encoding="utf-8"))

    def call():
        import requests
        os.makedirs("data/audio", exist_ok=True)
        wav = f"data/audio/{os.path.splitext(os.path.basename(video_path))[0]}.wav"
        subprocess.run([_ffmpeg(), "-y", "-v", "error", "-i", video_path, "-vn",
                        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
                        "-map_metadata", "-1", wav], check=True)
        settings = {"mode": "ENDPOINTING", "model": ASR_MODEL, "audioEncoding": "WAV"}
        with open(wav, "rb") as f:
            r = requests.post(f"{BASE_URL}/asr/transcribe",
                              headers={"Authorization": f"Bearer {os.environ['MODEL_API_KEY']}"},
                              files={"request": (None, json.dumps(settings), "application/json"),
                                     "audio": ("audio.wav", f, "audio/wav")},
                              timeout=300)
        r.raise_for_status()
        turns = r.json().get("turns", [])
        return [{"start": t["startMs"] / 1000, "end": t["endMs"] / 1000,
                 "text": t["transcript"].strip()} for t in turns if t["transcript"].strip()]

    return _cached("asr", _key(video_path), call)

# ---------- 2. SAM 3.1: find an object in an image ----------

BOX_RE = re.compile(r"<\|box;([^|]*)\|>")

def _parse_boxes(text):
    boxes = []
    for m in BOX_RE.finditer(text or ""):
        f = dict(kv.split("=", 1) for kv in m.group(1).split(";") if "=" in kv)
        boxes.append({k: float(f[k]) for k in ("x1", "y1", "x2", "y2", "w", "h") if k in f}
                     | {"score": float(f.get("c", 1.0))})
    return boxes

def segment(image_path, prompt):
    """Returns boxes [{"x1","y1","x2","y2","w","h","score"}] in image pixels."""
    if BACKEND == "mock":
        return []

    def call():
        resp = _client().responses.create(
            model=SAM_MODEL,
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": _data_url(image_path)},
            ]}],
            metadata={"include_confidence": "true"},
        )
        return {"raw": resp.output_text, "boxes": _parse_boxes(resp.output_text)}

    return _cached("sam", _key(image_path, prompt), call)["boxes"]

# ---------- 3. Muse Spark: reasoning over text + images ----------

def _spark(content, max_output_tokens=4000):
    resp = _client().responses.create(
        model=SPARK_MODEL,
        input=[{"type": "message", "role": "user", "content": content}],
        max_output_tokens=max_output_tokens,
    )
    _log_usage(SPARK_MODEL, resp)
    return resp.output_text

def _json_from(text):
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    if not m:
        raise ValueError(f"No JSON in model output:\n{text[:500]}")
    return json.loads(m.group(0))

SOP_PROMPT = """You are writing a Standard Operating Procedure (SOP) for a hands-on task.
You get, for each step: the worker's narration and one photo from the moment
the step was done. Use BOTH: the narration may be late, vague or wrong; trust
what the photo shows when they disagree.

Return ONLY JSON:
{"title": str, "purpose": str,
 "tools": [str], "parts": [str],
 "steps": [{"number": int, "title": str (max 6 words, imperative),
            "instruction": str (1-2 clear sentences),
            "visible_objects": [str] (2-3 short noun phrases of objects visible in the photo
                                      that matter for this step, e.g. "battery", "battery cover"),
            "check": str (how the worker confirms the step is done),
            "warning": str or null}]}
Keep the same number of steps and order as given."""

def write_sop(steps):
    """steps: [{"number","text","keyframe"}] -> SOP dict."""
    if BACKEND == "mock":
        return _mock_sop(steps)

    content = [{"type": "input_text", "text": SOP_PROMPT}]
    for s in steps:
        content.append({"type": "input_text",
                        "text": f"Step {s['number']} narration: {s.get('text') or '(no narration)'}"})
        content.append({"type": "input_image", "image_url": _data_url(s["keyframe"])})

    def call():
        return _json_from(_spark(content))

    key = _key(SPARK_MODEL, SOP_PROMPT, [(s["number"], s.get("text")) for s in steps],
               *[s["keyframe"] for s in steps])
    return _cached("sop", key, call)

CHECK_PROMPT = """You are checking a worker against an SOP. Below are the SOP steps,
then photos from ONE segment of a new recording (in time order).
Which SOP step does this segment show? Return ONLY JSON:
{"step": int or null (null if it matches no step), "confidence": number 0-1,
 "reason": str (one short sentence about what is visible)}"""

def match_step(sop, image_paths):
    """Which SOP step does a segment (a few frames) show?"""
    if BACKEND == "mock":
        return {"step": None, "confidence": 0.0, "reason": "mock mode"}

    sop_text = "\n".join(f"{s['number']}. {s['title']}: {s['instruction']}" for s in sop["steps"])
    content = [{"type": "input_text", "text": CHECK_PROMPT + "\n\nSOP:\n" + sop_text}]
    content += [{"type": "input_image", "image_url": _data_url(p)} for p in image_paths]

    def call():
        return _json_from(_spark(content, 500))

    return _cached("match", _key(SPARK_MODEL, CHECK_PROMPT, sop_text, *image_paths), call)

# ---------- mock SOP ----------

def _mock_sop(steps):
    out = []
    for s in steps:
        text = re.sub(r"^step \w+,?\s*", "", (s.get("text") or "Do the step").strip(), flags=re.I)
        text = re.sub(r"^i\s+", "", text, flags=re.I).rstrip(".")
        out.append({"number": s["number"], "title": text[:1].upper() + text[1:],
                    "instruction": text[:1].upper() + text[1:] + ".",
                    "visible_objects": [], "check": "Visually confirm the step is complete.",
                    "warning": None})
    return {"title": "Procedure (mock)", "purpose": "Generated in mock mode.",
            "tools": [], "parts": [], "steps": out}
