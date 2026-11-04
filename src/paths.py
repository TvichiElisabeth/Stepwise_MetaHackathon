"""Where every file lives. Paths are relative to the repo root.

  data/videos/          input recordings
  data/mock/            test data: fake transcripts, ground truth
  data/work/<name>/     temporary files, git-ignored, safe to delete
  data/output/<name>/   final results: open sop.html here
"""

def video(name):           return f"data/videos/{name}.mp4"
def mock_transcript(name): return f"data/mock/transcript_{name}.json"
def ground_truth(name):    return f"data/mock/ground_truth_{name}.json"

def work(name):            return f"data/work/{name}"
def frames_dir(name):      return f"{work(name)}/frames"
def keyframes_dir(name):   return f"{work(name)}/keyframes"
def plots_dir(name):       return f"{work(name)}/plots"
def audio(name):           return f"{work(name)}/audio.wav"
def transcript(name):      return f"{work(name)}/transcript.json"
def steps_json(name):      return f"{work(name)}/steps.json"
def preview(name):         return f"{work(name)}/preview.jpg"

def output(name):          return f"data/output/{name}"
def output_images(name):   return f"{output(name)}/images"
def sop_json(name):        return f"{output(name)}/sop.json"
def sop_html(name):        return f"{output(name)}/sop.html"
