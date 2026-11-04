"""Stepwise demo app.

  View SOP    shows saved results from data/output/. No API calls, no key needed.
  Create SOP  runs the real pipeline. Only appears when MODEL_API_KEY is set,
              so a public copy of the app can't spend credits.

Colours and fonts are in .streamlit/config.toml, the rest of the styling in CSS below.

Usage:  streamlit run src\\app.py
"""
import glob, html, json, os, re
import streamlit as st
from dotenv import load_dotenv
import paths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    ("Muse Voice Transcribe", "Turns the narration into timestamped text."),
    ("Muse Spark", "Groups the narration into steps, reads the photos and writes the SOP."),
    ("SAM 3.1", "Finds the objects Spark named and boxes them on each photo."),
]

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=JetBrains+Mono:wght@500;600&family=Space+Grotesk:wght@600;700&display=swap');
.block-container { max-width: 1120px; padding-top: 2.5rem; }
.sw-label { font: 600 11px/16px 'JetBrains Mono', monospace; letter-spacing: .08em; text-transform: uppercase; color: #565e74; }
.sw-title { font: 700 40px/48px 'Space Grotesk', sans-serif; letter-spacing: -.02em; text-transform: uppercase; margin: 0; color: #0b1c30; }
.sw-tagline { font-size: 16px; font-weight: 600; margin: 4px 0 28px; }
.sw-cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 8px 0 32px; }
.sw-card { background: #fff; padding: 16px; box-shadow: 0 1px 3px rgba(11,28,48,.1); }
.sw-card .sw-label { color: #8d4b00; }
.sw-card h3 { font: 600 18px/24px 'IBM Plex Sans', sans-serif; margin: 4px 0; padding: 0; color: #0b1c30; }
.sw-card p { font-size: 14px; line-height: 20px; color: #565e74; margin: 0; }
.sw-sop-title { font: 700 32px/40px 'Space Grotesk', sans-serif; letter-spacing: -.02em; text-transform: uppercase; margin: 4px 0 16px; color: #0b1c30; }
.sw-meta { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin-bottom: 24px; }
.sw-meta > div { background: #fff; padding: 12px 16px; box-shadow: 0 1px 3px rgba(11,28,48,.1); }
.sw-meta p { margin: 4px 0 0; }
.sw-step-head { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
.sw-num { background: #0b1c30; color: #fff; font: 600 12px/16px 'JetBrains Mono', monospace; padding: 3px 8px; }
.sw-step-title { font: 600 20px/28px 'Space Grotesk', sans-serif; color: #0b1c30; }
.sw-instruction { font-size: 16px; line-height: 24px; font-weight: 600; margin: 4px 0 12px; }
.sw-box { padding: 10px 14px; margin-bottom: 10px; font-size: 14px; line-height: 20px; font-weight: 600; }
.sw-box .sw-label { display: block; margin-bottom: 2px; }
.sw-check { background: #e3f4e7; color: #14532d; } .sw-check .sw-label { color: #1d7a35; }
.sw-warn { background: #ffdcc3; color: #6e3900; } .sw-warn .sw-label { color: #8d4b00; }
.sw-source { background: #eff4ff; color: #565e74; padding: 8px 12px; font-size: 12px; line-height: 16px; }
.sw-found { background: #eff4ff; padding: 6px 10px; font: 500 12px/16px 'JetBrains Mono', monospace; color: #0b1c30; }
.sw-found b { color: #8d4b00; }
[class*="st-key-step_"] { background: #fff; border: none !important; box-shadow: 0 2px 8px rgba(11,28,48,.1); padding: 20px; margin-bottom: 8px; }
[data-testid="stTab"] { background: #eff4ff; padding: 8px 20px; margin-right: 4px; }
[data-testid="stTab"] p { font: 600 12px/16px 'JetBrains Mono', monospace; letter-spacing: .06em; text-transform: uppercase; color: #565e74; }
[data-testid="stTab"][aria-selected="true"] { background: #0b1c30; }
[data-testid="stTab"][aria-selected="true"] p { color: #fff; }
@media (max-width: 700px) { .sw-cards, .sw-meta { grid-template-columns: 1fr; } .sw-title { font-size: 30px; line-height: 36px; } .sw-sop-title { font-size: 24px; line-height: 32px; } }
</style>"""

def show_html(markup):
    st.markdown(markup, unsafe_allow_html=True)

def e(text):
    """Escapes model output for HTML. A blank line would end the HTML block, so whitespace is collapsed."""
    return html.escape(" ".join(str(text).split()))

def saved_sops():
    return sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob("data/output/*/sop.json"))

def show_step(step, key):
    with st.container(border=True, key=key):
        show_html(f'<div class="sw-step-head"><span class="sw-num">{step["number"]:02d}</span>'
                  f'<span class="sw-step-title">{e(step["title"])}</span></div>')
        text, photo = st.columns([5, 7], gap="medium")
        with text:
            parts = ['<div class="sw-label">Instruction</div>',
                     f'<p class="sw-instruction">{e(step["instruction"])}</p>']
            if step.get("check"):
                parts.append(f'<div class="sw-box sw-check"><span class="sw-label">Check</span>{e(step["check"])}</div>')
            if step.get("warning"):
                parts.append(f'<div class="sw-box sw-warn"><span class="sw-label">Warning</span>{e(step["warning"])}</div>')
            source = []
            if "start" in step:
                source.append(f"Video {step['start']:.1f}–{step['end']:.1f} s.")
            if step.get("narration"):
                source.append(f"Narration: “{e(step['narration'])}”")
            if source:
                parts.append(f'<div class="sw-source">{" ".join(source)}</div>')
            show_html("".join(parts))
        with photo:
            if step.get("image") and os.path.exists(step["image"]):
                st.image(step["image"], width="stretch")
            found = ", ".join(f"<b>{e(o['object'])} ({o['score']:.2f})</b>"
                              for o in step.get("found_objects", []))
            show_html(f'<div class="sw-found">SAM 3.1 found: {found}</div>' if found
                      else '<div class="sw-found">SAM 3.1 found no objects here</div>')

def show_sop(name, where, download_slot=None):
    """where: keeps widget keys unique, since both tabs can show an SOP in the same run."""
    sop = json.load(open(paths.sop_json(name), encoding="utf-8"))
    if os.path.exists(paths.sop_html(name)):
        with open(paths.sop_html(name), "rb") as f:
            (download_slot or st).download_button("Download SOP as HTML", f.read(), key=f"dl_{where}_{name}",
                                                  file_name=f"{name}_sop.html", mime="text/html")
    meta = f'<div><div class="sw-label">Purpose</div><p>{e(sop.get("purpose", ""))}</p></div>'
    lists = "".join(f'<div class="sw-label">{label}</div><p>{e(", ".join(sop[k]))}</p>'
                    for label, k in (("Tools", "tools"), ("Parts", "parts")) if sop.get(k))
    show_html(f'<h2 class="sw-sop-title">{e(sop["title"])}</h2>'
              f'<div class="sw-meta">{meta}<div>{lists}</div></div>')
    for step in sop["steps"]:
        show_step(step, f"step_{where}_{name}_{step['number']}")

def view_tab():
    names = saved_sops()
    if not names:
        st.info("No saved SOPs yet.")
        return
    pick, download = st.columns([3, 1], vertical_alignment="bottom")
    name = pick.selectbox("Saved SOP", names)
    show_sop(name, "view", download)

def create_tab():
    st.write("Upload a narrated video filmed from a fixed camera, with a short pause between steps.")
    st.caption("This calls the Meta Model API and uses credits. Repeat runs of the same video are "
               "served from the cache. The bottom 16% of each frame is cropped.")
    upload = st.file_uploader("Narrated video (.mp4)", type=["mp4"])
    name = st.text_input("Short name, for example ram_swap").strip().lower()

    if not st.button("Create SOP", type="primary", disabled=not (upload and name)):
        return
    if not re.fullmatch(r"[a-z0-9_-]+", name):
        st.error("Use only lowercase letters, numbers, _ and -.")
        return
    if os.path.exists(paths.video(name)) or os.path.exists(paths.work(name)):
        st.error(f"“{name}” is already used. Pick another name.")
        return

    os.makedirs(os.path.dirname(paths.video(name)), exist_ok=True)
    with open(paths.video(name), "wb") as f:
        f.write(upload.getbuffer())
    os.makedirs(paths.output(name), exist_ok=True)

    from sop import create_sop      # imported here: viewing saved SOPs doesn't need OpenCV
    with st.status("Creating SOP...", expanded=True) as status:
        try:
            create_sop(name, log=st.write)
        except Exception as err:
            status.update(label="Failed", state="error")
            st.error(f"{type(err).__name__}: {err}")
            return
        status.update(label="SOP created", state="complete")
    show_sop(name, "create")

def main():
    os.chdir(ROOT)                  # paths.py is relative to the repo root
    load_dotenv()
    st.set_page_config(page_title="Stepwise", layout="wide")
    show_html(CSS)
    show_html('<h1 class="sw-title">Stepwise</h1>'
              '<p class="sw-tagline">Turns a narrated video of a hands-on task into a Standard Operating Procedure.</p>')
    cards = "".join(f'<div class="sw-card"><div class="sw-label">Stage {i}</div><h3>{model}</h3><p>{job}</p></div>'
                    for i, (model, job) in enumerate(MODELS, 1))
    show_html(f'<div class="sw-label">How it works</div><div class="sw-cards">{cards}</div>')

    if os.getenv("MODEL_API_KEY"):
        view, create = st.tabs(["View SOP", "Create SOP"])
        with view:
            view_tab()
        with create:
            create_tab()
    else:
        view_tab()

if __name__ == "__main__":
    main()
