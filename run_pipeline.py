# run_pipeline.py
# Teacher-style annotation pipeline:
#   background.pdf -> background.png
#   narration.wav  -> whisper transcript -> annotation planner -> Pillow renderer -> moviepy -> output.mp4

import sys
import os
import json

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Gemini API ─────────────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# ── Make ffmpeg available to Whisper (and MoviePy) via imageio-ffmpeg ──────────
import imageio_ffmpeg
_ffmpeg_exe  = imageio_ffmpeg.get_ffmpeg_exe()          # full path to binary
_ffmpeg_dir  = str(Path(_ffmpeg_exe).parent)            # directory containing it
os.environ["IMAGEIO_FFMPEG_EXE"] = _ffmpeg_exe
os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
SRC_DIR      = PROJECT_ROOT / "src"
OUTPUT_DIR   = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: PDF → PNG ──────────────────────────────────────────────────────────
print("[1/5] Converting PDF to PNG...")
from pdf2image import convert_from_path

pdf_path = DATA_DIR / "background.pdf"
pages    = convert_from_path(str(pdf_path), first_page=1, last_page=1)
bg_pil   = pages[0]
W, H     = bg_pil.size

bg_png_path = SRC_DIR / "background.png"
bg_pil.save(str(bg_png_path), "PNG")
print(f"    Background size: {W}x{H}")

# Ensure dimensions are divisible by 2 for libx264 (avoid yuv444p issues)
if W % 2 == 1 or H % 2 == 1:
    new_W = W + (W % 2)
    new_H = H + (H % 2)
    new_bg = Image.new("RGB", (new_W, new_H), (255, 255, 255))
    new_bg.paste(bg_pil, (0, 0))
    bg_pil = new_bg
    W, H = bg_pil.size
    bg_pil.save(str(bg_png_path), "PNG")
    print(f"    Padded background to even dimensions: {W}x{H}")

# ── Step 2: Audio duration ─────────────────────────────────────────────────────
print("[2/5] Loading audio...")
# pyrefly: ignore [missing-import]
from moviepy.editor import AudioFileClip, ImageClip, CompositeVideoClip, VideoClip
import subprocess
import tempfile
import scipy.signal
import scipy.io.wavfile

# The source file may be an MP3 with a .wav extension (ID3 header).
# Use ffmpeg to transcode it into a genuine 16 kHz mono PCM WAV in a temp file.
audio_path = DATA_DIR / "narration.wav"
print(f"    Audio file: {audio_path}")

_tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
_tmp_wav.close()
print(f"    Transcoding to: {_tmp_wav.name}")
subprocess.run(
    [_ffmpeg_exe, "-y", "-i", str(audio_path),
     "-ar", "16000", "-ac", "1", "-f", "wav", _tmp_wav.name],
    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)

# Load audio clip from transcoded temp WAV
audio_clip = AudioFileClip(_tmp_wav.name)
print(audio_clip.duration)
print(audio_clip.fps)
duration   = audio_clip.duration
print(f"    Duration: {duration:.2f}s")

# ── Step 3: Whisper transcription ──────────────────────────────────────────────
print("[3/5] Running Whisper transcription (this may take a minute)...")
import whisper

# Read the transcoded temp WAV for Whisper
_sr, _raw_audio = scipy.io.wavfile.read(_tmp_wav.name)

# Convert to float32 in [-1, 1]
if _raw_audio.dtype == np.int16:
    _raw_audio = _raw_audio.astype(np.float32) / 32768.0
elif _raw_audio.dtype == np.int32:
    _raw_audio = _raw_audio.astype(np.float32) / 2147483648.0
else:
    _raw_audio = _raw_audio.astype(np.float32)

# Ensure mono
if _raw_audio.ndim == 2:
    _raw_audio = _raw_audio.mean(axis=1)

_raw_audio = _raw_audio.astype(np.float32)
print(f"    Audio decoded: {len(_raw_audio)/16000:.2f}s at 16 kHz")

model    = whisper.load_model("tiny")          # fastest; swap to "base" for accuracy
result   = model.transcribe(_raw_audio, word_timestamps=False)
segments = result.get("segments", [])

# Fallback: if Whisper returns nothing, create evenly spaced dummy segments
if not segments:
    print("    WARNING: Whisper returned no segments; using fallback.")
    seg_len = 5.0
    n       = max(1, int(duration // seg_len))
    segments = [
        {"start": i * seg_len,
         "end":   min((i + 1) * seg_len, duration),
         "text":  ""}
        for i in range(n)
    ]

print(f"    Got {len(segments)} segments.")

# ── SAVE transcript.json ──────────────────────────────────────────────────────
full_transcript = result.get("text", "").strip()
transcript_data = {
    "text": full_transcript,
    "duration": duration,
    "segments": segments,
    "num_segments": len(segments)
}
transcript_json_path = OUTPUT_DIR / "transcript.json"
with open(transcript_json_path, "w", encoding="utf-8") as f:
    json.dump(transcript_data, f, indent=2)
print(f"    [SAVED] Transcript: {transcript_json_path}")

# ── Step 4: Annotation planner ────────────────────────────────────────────────
print("[4/5] Planning annotations...")

transcript_excerpt = (full_transcript[:200] + "...") if len(full_transcript) > 200 else full_transcript
print(f"    Transcript excerpt: {transcript_excerpt}")

# Variable to track raw Gemini annotations (before coordinate conversion)
annotations_raw_gemini = None
annotations_json_path = None  # Initialize to None

# Region → coordinate mapping
REGION_COORDS = {
    "question_area": (0.25, 0.30),   # top-left quadrant
    "diagram_area": (0.70, 0.30),    # top-right quadrant
    "working_area": (0.50, 0.65),    # bottom-center
}

def region_to_coords(region_name):
    """Convert region name to pixel coordinates."""
    if region_name not in REGION_COORDS:
        region_name = "working_area"
    ax, ay = REGION_COORDS[region_name]
    return int(ax * W), int(ay * H)

def plan_annotations_with_gemini(transcript, ocr_text, duration, segments):
    """Use Gemini Flash to generate raw annotations."""
    print(f"    [DEBUG] OCR text length: {len(ocr_text)}")
    ocr_sample = (ocr_text[:300] + "...") if len(ocr_text) > 300 else ocr_text
    print(f"    [DEBUG] OCR text sample: {ocr_sample}")
    print("    [DEBUG] Confirmation: OCR text is included in the Gemini prompt.")

    if not HAS_GEMINI:
        return None
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("models/gemini-2.5-flash")

        prompt = f"""You are an experienced teacher creating educational annotations for any academic subject.

Your task: Extract CONCRETE EDUCATIONAL CONTENT from BOTH the OCR content (what is visible) and the Transcript (what is spoken) to create annotations that match what a teacher would write on a board.

VISIBLE CONTENT (OCR):
{ocr_text}

SPOKEN EXPLANATION (TRANSCRIPT) (duration {duration:.1f} seconds):
{transcript}

CRITICAL RULES:

1. COVER ALL IMPORTANT EDUCATIONAL ELEMENTS. Do not generate only formulas! You must generate annotations for:
   - Question statement
   - Given values
   - Options (if MCQ)
   - Key definitions
   - Formulas
   - Substitution steps
   - Intermediate calculations
   - Final answer
   - Correct option

2. EXTRACT, DO NOT GENERATE generic labels:
   ❌ NEVER create: "Distance Formula", "Intermediate steps", "Final Answer", "Key Concept"
   ✅ ALWAYS create: Actual content like "Find the distance between the points A (1, 2) and (4,6)."
   ✅ ALWAYS create: Actual formulas like "d = √((x₂-x₁)² + (y₂-y₁)²)"
   ✅ ALWAYS create: Actual substitutions like "√((4-1)² + (6-2)²)"
   ✅ ALWAYS create: Actual values like "5 units", "(C) 5 units", "x = 3"

3. SOURCE MATERIAL:
   - Extract content from BOTH the OCR and the Transcript.
   - Content should look exactly like what the teacher says or writes, or what is in the question text/options.

4. EVENT TYPES:
   - show_text: Use this for formulas, steps, values, options, given values.
   - highlight: Highlight when teacher mentions the question/problem statement.
   - draw_arrow: Point when teacher refers to figures, diagrams, or visual objects.

5. GENERATE ALL INTERMEDIATE SIMPLIFICATION STEPS:
   - Carefully follow the teacher's mathematical calculations.
   - You must output all intermediate simplification and calculation steps as separate events (e.g. if the teacher simplifies √((4-1)² + (6-2)²) to √(3² + 4²), and then to √(9 + 16), write a step for each). Do not skip these intermediate steps even if they seem trivial.

EVENT SCHEMA:
{{
  "category": "question" | "given_values" | "options" | "formula" | "substitution" | "calculation" | "answer" | "correct_option",
  "source": "ocr" | "transcript",
  "trigger_quote": "<exact spoken phrase from transcript if source is transcript, else empty>",
  "type": "highlight" | "show_text" | "draw_arrow",
  "region": "question_area" | "diagram_area" | "working_area",
  "content": <string>
}}

EXAMPLE OF CORRECT OUTPUT:
[
  {{"category": "question", "source": "ocr", "trigger_quote": "", "type": "highlight", "region": "question_area", "content": "Find the distance between the points (1, 2) and (4, 6)"}},
  {{"category": "given_values", "source": "ocr", "trigger_quote": "", "type": "show_text", "region": "question_area", "content": "A (1, 2) and (4,6)"}},
  {{"category": "options", "source": "ocr", "trigger_quote": "", "type": "show_text", "region": "question_area", "content": "(A) 3 units\\n(B) 4 units\\n(C) 5 units\\n(D) 6 units"}},
  {{"category": "formula", "source": "transcript", "trigger_quote": "distance between 2 points is given as", "type": "show_text", "region": "working_area", "content": "d = √((x₂-x₁)² + (y₂-y₁)²)"}},
  {{"category": "substitution", "source": "transcript", "trigger_quote": "we will write as 4 minus 1 whole square plus y2 is 6, so we will write as 6 minus 2 whole square", "type": "show_text", "region": "working_area", "content": "d = √((4-1)² + (6-2)²)"}},
  {{"category": "calculation", "source": "transcript", "trigger_quote": "under root of x2 plus 4 square", "type": "show_text", "region": "working_area", "content": "d = √(3² + 4²)"}},
  {{"category": "calculation", "source": "transcript", "trigger_quote": "under root of 9 plus 16", "type": "show_text", "region": "working_area", "content": "d = √(9 + 16)"}},
  {{"category": "calculation", "source": "transcript", "trigger_quote": "comes out to be under root 25", "type": "show_text", "region": "working_area", "content": "d = √25"}},
  {{"category": "answer", "source": "transcript", "trigger_quote": "answer will be d is equal to 5 units", "type": "show_text", "region": "working_area", "content": "d = 5 units"}},
  {{"category": "correct_option", "source": "ocr", "trigger_quote": "", "type": "show_text", "region": "question_area", "content": "(C) 5 units"}}
]

Generate 5-12 meaningful annotations. Ensure "content" is always provided.
Return ONLY valid JSON array, nothing else."""

        response = model.generate_content(prompt)
        
        print("\n===== RAW GEMINI RESPONSE =====")
        print(response.text)
        print("===============================\n")
        
        response_text = response.text.strip()
        
        # Extract JSON from response
        start_idx = response_text.find("[")
        end_idx = response_text.rfind("]") + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            annotations_raw = json.loads(json_str)
            return annotations_raw
            
    except Exception as e:
        print(f"    ⚠ Gemini planner error: {e}")
        
    return None

def map_raw_annotations_to_timeline(annotations_raw, duration, segments, ocr_data):
    """Convert raw annotations (category, source, trigger_quote, etc.) to mapped timeline events."""
    import string
    
    def normalize_trigger(text):
        if not text: return ""
        t = text.lower()
        for p in string.punctuation:
            t = t.replace(p, " ")
        return " ".join(t.split())

    # Pre-build concatenated transcript and char-to-segment mapping
    concat_text = ""
    char_to_segment = []

    for i, seg in enumerate(segments):
        seg_norm = normalize_trigger(seg.get("text", ""))
        if not seg_norm:
            continue
        if concat_text:
            concat_text += " "
            char_to_segment.append(None)
        
        seg_char_start = len(concat_text)
        seg_char_len = len(seg_norm)
        concat_text += seg_norm
        for char_idx in range(seg_char_start, len(concat_text)):
            char_to_segment.append((i, seg, seg_char_start, seg_char_len))

    def get_char_time(char_idx):
        entry = char_to_segment[char_idx]
        if not entry:
            left = char_idx
            while left >= 0 and char_to_segment[left] is None:
                left -= 1
            right = char_idx
            while right < len(char_to_segment) and char_to_segment[right] is None:
                right += 1
            if right < len(char_to_segment):
                entry = right
            elif left >= 0:
                entry = left
            else:
                return 0.0, None
        
        entry = char_to_segment[char_idx] if char_to_segment[char_idx] else char_to_segment[entry]
        if not entry:
            return 0.0, None
        seg_i, seg, seg_char_start, seg_char_len = entry
        offset = char_idx - seg_char_start
        seg_dur = seg["end"] - seg["start"]
        
        if seg_char_len > 1:
            frac = offset / (seg_char_len - 1)
            frac = max(0.0, min(1.0, frac))
            t = seg["start"] + frac * seg_dur
        else:
            t = seg["start"]
        return t, seg_i

    def find_ocr_bbox(target_text):
        if not ocr_data or not target_text:
            return None
        t_norm = "".join(target_text.lower().split())
        for item in ocr_data:
            item_text = item.get("text", "")
            item_norm = "".join(item_text.lower().split())
            if len(item_norm) > 3 and (t_norm in item_norm or item_norm in t_norm):
                return item["bbox"]
        return None

    annotations = []
    debug_records = []
    
    # Expected categories check
    expected_categories = {"question", "given_values", "options", "formula", "substitution", "calculation", "answer", "correct_option"}
    generated_categories = {ann.get("category") for ann in annotations_raw}
    missing_categories = expected_categories - generated_categories
    if missing_categories:
        print(f"\n    [WARNING] Missing educational categories: {missing_categories}\n")

    for ann in annotations_raw:
        category = ann.get("category", "")
        region = ann.get("region", "working_area")
        content = ann.get("content", "").strip()
        source = ann.get("source", "transcript")
        trigger_quote = ann.get("trigger_quote", "").strip()
        
        # 1. OCR text must never be re-rendered if it already exists in the PDF.
        # 2. OCR should only be used for localization (highlight, arrow, answer selection).
        if source == "ocr" and category not in ["question", "correct_option"]:
            print(f"    [TIMELINE CLEANUP] Skipping redundant OCR overlay: {category} - '{content}'")
            continue
            
        if category == "question":
            # Highlight question statement area over the entire video
            start_t = 0.0
            end_t = duration
            bbox = find_ocr_bbox(content)
            if bbox:
                x0 = min(p[0] for p in bbox)
                y0 = min(p[1] for p in bbox)
                x1 = max(p[0] for p in bbox)
                y1 = max(p[1] for p in bbox)
            else:
                x0, y0, x1, y1 = 54, 133, 1873, 238
                
            event = {
                "category": category,
                "type": "highlight",
                "start": start_t,
                "end": end_t,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1
            }
            debug_records.append({
                "category": category,
                "content": content,
                "trigger_quote": "*(None: OCR source)*",
                "matched_segment": "*(Full duration background anchor)*",
                "start": start_t,
                "end": end_t
            })
        elif category == "correct_option":
            # Correct option highlights the correct answer in green
            start_t = None
            end_t = None
            matched_seg_txt = ""
            for seg in reversed(segments):
                seg_txt = seg.get("text", "").lower()
                if "option" in seg_txt or "answer" in seg_txt or "c" in seg_txt:
                    start_t = seg["start"]
                    end_t = seg["end"]
                    matched_seg_txt = seg.get("text", "").strip()
                    break
            if start_t is None:
                start_t = duration - 5.0
                end_t = duration
                matched_seg_txt = "(Fallback: last 5 seconds)"
                
            bbox = find_ocr_bbox(content)
            if bbox:
                x0 = min(p[0] for p in bbox)
                y0 = min(p[1] for p in bbox)
                x1 = max(p[0] for p in bbox)
                y1 = max(p[1] for p in bbox)
            else:
                x0, y0, x1, y1 = 60, 520, 390, 580
                
            event = {
                "category": category,
                "type": "correct_option_highlight",
                "start": start_t,
                "end": end_t,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1
            }
            debug_records.append({
                "category": category,
                "content": content,
                "trigger_quote": "*(None: OCR source)*",
                "matched_segment": matched_seg_txt,
                "start": start_t,
                "end": end_t
            })
        else:
            # Transcript-based step annotation
            trigger_norm = normalize_trigger(trigger_quote)
            if not trigger_norm:
                print(f"    [MATCH WARNING] Skipping annotation due to empty trigger_quote for: {content}")
                continue
                
            idx = concat_text.find(trigger_norm)
            if idx == -1:
                print(f"    [MATCH WARNING] Skipping annotation because trigger_quote could not be matched: '{trigger_quote}' for content: '{content}'")
                continue
                
            start_idx = idx
            end_idx = idx + len(trigger_norm) - 1
            
            char_times = []
            for c_idx in range(start_idx, end_idx + 1):
                t, seg_i = get_char_time(c_idx)
                if seg_i is not None:
                    char_times.append((seg_i, t))
            
            seg_durations = {}
            for seg_i, t in char_times:
                if seg_i not in seg_durations:
                    seg_durations[seg_i] = []
                seg_durations[seg_i].append(t)
            
            kept_segs = []
            max_seg_i = None
            max_dur = -1.0
            
            for seg_i, times in seg_durations.items():
                dur = max(times) - min(times)
                if dur >= 1.0:
                    kept_segs.append(seg_i)
                if dur > max_dur:
                    max_dur = dur
                    max_seg_i = seg_i
            
            if not kept_segs and max_seg_i is not None:
                kept_segs = [max_seg_i]
            
            if kept_segs:
                kept_segs.sort()
                start_t = min(t for _, t in char_times)
                end_t = segments[kept_segs[-1]]["end"]
                matched_seg_txt = " ".join([segments[s]["text"].strip() for s in kept_segs])
                print(f"    [MATCH DEBUG] Gemini content: {content}")
                print(f"    [MATCH DEBUG] Trigger quote:  {trigger_quote}")
                print(f"    [MATCH DEBUG] Segment range:   {kept_segs[0]} - {kept_segs[-1]}")
                print(f"    [MATCH DEBUG] Timestamp:      {start_t:.2f} - {end_t:.2f}\n")
            else:
                print(f"    [MATCH WARNING] Skipping annotation because no segments could be mapped: '{trigger_quote}' for content: '{content}'")
                continue
            
            event = {
                "category": category,
                "type": "show_text",
                "start": start_t,
                "end": end_t,
                "content": content
            }
            debug_records.append({
                "category": category,
                "content": content,
                "trigger_quote": trigger_quote,
                "matched_segment": matched_seg_txt,
                "start": start_t,
                "end": end_t
            })
            
        annotations.append(event)
        
    # Print the debug table
    print("\n" + "="*160)
    print("TIMELINE DEBUG TABLE")
    print("="*160)
    print(f"{'CATEGORY':<15} | {'CONTENT':<40} | {'TRIGGER_QUOTE':<30} | {'MATCHED_SEGMENT':<40} | {'START':<6} | {'END':<6}")
    print("-"*160)
    for r in debug_records:
        cat = r["category"]
        cnt = r["content"].replace('\n', ' ')
        cnt_short = cnt[:37] + '...' if len(cnt) > 40 else cnt
        trg = r["trigger_quote"]
        trg_short = trg[:27] + '...' if len(trg) > 30 else trg
        seg = r["matched_segment"]
        seg_short = seg[:37] + '...' if len(seg) > 40 else seg
        start_val = f"{r['start']:.2f}"
        end_val = f"{r['end']:.2f}"
        print(f"{cat:<15} | {cnt_short:<40} | {trg_short:<30} | {seg_short:<40} | {start_val:<6} | {end_val:<6}")
    print("="*160 + "\n")
    
    return annotations

# Load OCR data locally
ocr_data = []
ocr_text = ""
ocr_json_path = OUTPUT_DIR / "ocr.json"
if ocr_json_path.exists():
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)
        ocr_text = "\n".join([item["text"] for item in ocr_data])

# ── Planning & Mapping ────────────────────────────────────────────────────────
annotations_raw = None
annotations_json_path = OUTPUT_DIR / "annotations.json"

# Try Gemini first
if HAS_GEMINI and os.environ.get("GOOGLE_API_KEY"):
    print("    Attempting to plan annotations with Gemini API...")
    annotations_raw = plan_annotations_with_gemini(full_transcript, ocr_text, duration, segments)
    if annotations_raw:
        # Save raw annotations to annotations.json
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["question", "given_values", "options", "formula", "substitution", "calculation", "answer", "correct_option"]},
                    "source": {"type": "string", "enum": ["ocr", "transcript"]},
                    "trigger_quote": {"type": "string"},
                    "type": {"type": "string", "enum": ["highlight", "show_text", "draw_arrow", "draw_equation"]},
                    "region": {"type": "string", "enum": ["question_area", "diagram_area", "working_area"]},
                    "content": {"type": "string"}
                },
                "required": ["category", "source", "type", "region", "content"]
            }
        }
        with open(annotations_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "schema": schema,
                "annotations": annotations_raw
            }, f, indent=2)
        print(f"    [SAVED] annotations.json: {annotations_json_path}")
        annotations_raw_gemini = annotations_raw

# If Gemini fails, load cached plan
if not annotations_raw:
    if annotations_json_path.exists():
        print(f"    [CACHE] Gemini API unavailable or rate-limited. Loading cached annotations from {annotations_json_path}...")
        try:
            with open(annotations_json_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                annotations_raw = cached_data.get("annotations", [])
                annotations_raw_gemini = annotations_raw
        except Exception as e:
            print(f"    ⚠ Error reading cached annotations: {e}")

# Map raw annotations to timeline if we have them
if annotations_raw:
    print("    Mapping raw annotations to timeline coordinates and timestamps...")
    annotations = map_raw_annotations_to_timeline(annotations_raw, duration, segments, ocr_data)
else:
    annotations = None

if annotations:
    print(f"    Planner generated {len(annotations)} timeline annotations.")
else:
    print("    No planner source (Gemini or Cache) available; falling back to keyword-based planner.")
    
    # Fallback: Keyword-based planner
    HIGHLIGHT_KEYWORDS = ["important", "note", "observe", "notice"]
    ARROW_KEYWORDS     = ["triangle", "diagram", "figure"]
    SHOW_TEXT_KEYWORDS = ["equation", "equals", "formula"]

    ANCHORS = [
        (0.25, 0.30),   # top-left quadrant
        (0.70, 0.30),   # top-right quadrant
        (0.25, 0.65),   # bottom-left quadrant
        (0.70, 0.65),   # bottom-right quadrant
        (0.50, 0.50),   # centre
    ]

    def pick_anchor(idx):
        """Rotate through anchors based on segment index."""
        ax, ay = ANCHORS[idx % len(ANCHORS)]
        return int(ax * W), int(ay * H)

    annotations = []

    def choose_segment_text_at(timestamp):
        best_text = None
        best_dist = float("inf")
        for seg in segments:
            seg_text = seg.get("text", "").strip()
            if not seg_text:
                continue
            start = seg["start"]
            end = seg["end"]
            if start <= timestamp < end:
                return seg_text
            midpoint = (start + end) / 2
            dist = abs(midpoint - timestamp)
            if dist < best_dist:
                best_dist = dist
                best_text = seg_text
        return best_text

    event_index = 0
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        text_lower = text.lower()
        start = seg["start"]
        end   = seg["end"]
        x, y  = pick_anchor(event_index)

        if any(k in text_lower for k in SHOW_TEXT_KEYWORDS):
            annotations.append({
                "type": "show_text", "start": start, "end": end,
                "x": x, "y": y, "content": text[:120]
            })
            event_index += 1
            continue

        if any(k in text_lower for k in HIGHLIGHT_KEYWORDS):
            annotations.append({
                "type": "highlight", "start": start, "end": end,
                "x": x, "y": y, "w": int(W * 0.35), "h": int(H * 0.08)
            })
            event_index += 1
            continue

        if any(k in text_lower for k in ARROW_KEYWORDS):
            annotations.append({
                "type": "draw_arrow", "start": start, "end": end,
                "x": x, "y": y
            })
            event_index += 1
            continue

    # Fallback: if no keyword-driven annotations were generated, create events every 8-12 seconds
    if not annotations:
        print("    No keyword-driven annotations found; generating fallback events.")
        interval = 10.0
        fallback_duration = 8.0
        num_events = max(1, int(np.ceil(duration / interval)))

        for idx in range(num_events):
            start = min(idx * interval, max(0.0, duration - fallback_duration))
            end = min(start + fallback_duration, duration)
            fallback_text = choose_segment_text_at(start) or full_transcript
            if not fallback_text:
                continue

            content = fallback_text.strip()[:120]
            if not content:
                continue

            x, y = pick_anchor(idx)
            annotations.append({
                "type": "show_text", "start": start, "end": end,
                "x": x, "y": y, "content": content
            })

# Save timeline to OUTPUT directory (final renderer-ready)
timeline_path = OUTPUT_DIR / "annotation_timeline.json"
with open(timeline_path, "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=2)
print(f"    Generated {len(annotations)} annotation events.")
print(f"    [SAVED] Timeline: {timeline_path}")
print(f"    Transcript excerpt: {transcript_excerpt}")
print(json.dumps(annotations[:5], indent=2), "..." if len(annotations) > 5 else "")

# ── Step 5: Pillow-based renderer → MoviePy ────────────────────────────────────
print("[5/5] Rendering video with annotations...")

bg_arr = np.array(bg_pil.convert("RGB"))   # (H, W, 3) uint8

# Try to load handwriting font; fall back to default if unavailable
try:
    font_path = str(PROJECT_ROOT / "assets" / "fonts" / "Kalam-Regular.ttf")
    if not os.path.exists(font_path):
        font_path = str(PROJECT_ROOT / "assets" / "fonts" / "PatrickHand-Regular.ttf")
    FONT_HAND = ImageFont.truetype(font_path, size=max(64, H // 22))
except Exception as e:
    print(f"Warning: Handwriting font not found, falling back. ({e})")
    FONT_HAND = ImageFont.load_default()

def get_wobbly_line(p_start, p_end, fraction, seed_val):
    x_s, y_s = p_start
    x_e, y_e = p_end
    
    # Calculate target end point based on fraction
    x_target = x_s + fraction * (x_e - x_s)
    y_target = y_s + fraction * (y_e - y_s)
    
    dist = ((x_e - x_s)**2 + (y_e - y_s)**2)**0.5
    if dist == 0:
        return []
        
    rng = random.Random(seed_val)
    pts = []
    
    num_steps = int(dist / 20) + 1
    for step in range(num_steps + 1):
        step_frac = step / num_steps
        if step_frac > fraction:
            break
        curr_x = x_s + step_frac * (x_e - x_s)
        curr_y = y_s + step_frac * (y_e - y_s)
        
        # Add perpendicular jitter (±2 px)
        dx = x_e - x_s
        dy = y_e - y_s
        perp_x = -dy / dist
        perp_y = dx / dist
        jitter = rng.randint(-2, 2)
        
        pts.append((curr_x + perp_x * jitter, curr_y + perp_y * jitter))
        
    # Always end precisely at the target point (with some jitter)
    dx = x_e - x_s
    dy = y_e - y_s
    perp_x = -dy / dist
    perp_y = dx / dist
    jitter = rng.randint(-2, 2)
    pts.append((x_target + perp_x * jitter, y_target + perp_y * jitter))
    
    return pts

def draw_frame(t):
    """Return a numpy RGB frame at time t with all active annotations drawn."""
    # Start from a fresh copy of the background
    img  = Image.fromarray(bg_arr.copy())
    draw = ImageDraw.Draw(img, "RGBA")

    # 1. First, draw all highlights (e.g. question highlight and correct answer highlight)
    for ann in annotations:
        ann_type = ann.get("type")
        if ann_type not in ["highlight", "correct_option_highlight", "draw_arrow"]:
            continue
            
        if not (ann["start"] <= t < ann["end"]):
            continue

        # Draw highlight
        if ann_type == "highlight":
            x0 = ann.get("x0")
            y0 = ann.get("y0")
            x1 = ann.get("x1")
            y1 = ann.get("y1")
            if x0 is not None and y0 is not None and x1 is not None and y1 is not None:
                bbox_width = x1 - x0
                bbox_height = y1 - y0
                underline_duration = 7.0
                fraction = min(1.0, (t - ann["start"]) / underline_duration)
                if fraction > 0:
                    y_pos = y1 + int(bbox_height * 0.06)
                    p_start = (x0, y_pos)
                    p_end = (x1, y_pos)
                    pts = get_wobbly_line(p_start, p_end, fraction, 999)
                    if len(pts) >= 2:
                        thickness = max(3, int(bbox_height * 0.045))
                        draw.line(pts, fill=(25, 35, 90, 255), width=thickness)
                        
        elif ann_type == "correct_option_highlight":
            x0 = ann.get("x0")
            y0 = ann.get("y0")
            x1 = ann.get("x1")
            y1 = ann.get("y1")
            if x0 is not None and y0 is not None and x1 is not None and y1 is not None:
                # Fade in effect: 0.3s fade duration
                time_in = t - ann["start"]
                fade_duration = 0.3
                alpha_factor = min(1.0, time_in / fade_duration)
                
                # Soft transparent green outline box for selected answer
                fill_color = (100, 255, 100, int(50 * alpha_factor))
                outline_color = (40, 190, 40, int(220 * alpha_factor))
                
                draw.rectangle([x0, y0, x1, y1], fill=fill_color, outline=outline_color, width=6)
                
        elif ann_type == "draw_arrow":
            x = ann.get("x", W // 2)
            y = ann.get("y", H // 2)
            ax, ay = x, y
            tx, ty = ax - int(W * 0.15), ay - int(H * 0.15)
            arrow_width = max(10, W // 60)
            draw.line([tx, ty, ax, ay], fill=(255, 0, 0, 255), width=arrow_width)
            angle_offset = int(W * 0.04)
            draw.polygon([ax, ay, ax - angle_offset, ay - angle_offset // 2, ax - angle_offset // 2, ax - angle_offset], fill=(255, 0, 0, 255))

    # 2. Gather all active working steps (transcript show_text annotations that have started)
    active_steps = []
    for ann in annotations:
        if ann.get("type") == "show_text" and ann.get("start") <= t:
            # Skip OCR-derived content (only render transcript solving steps)
            if ann.get("category") not in ["formula", "substitution", "calculation", "answer"]:
                continue
                
            # Avoid duplicate content if any
            content = ann.get("content", "").strip()
            
            # Skip optional 3D-distance formulas for 2D coordinate geometry problems
            if "3D" in content or "z\u2082" in content or "z2" in content:
                continue
                
            if content and content not in [s["content"] for s in active_steps]:
                active_steps.append({
                    "content": content,
                    "start": ann["start"],
                    "category": ann.get("category", "")
                })
                
    # Sort active steps chronologically by start time
    active_steps.sort(key=lambda s: round(s["start"], 2))

    # 3. Draw solution steps with a handwriting typing effect
    if active_steps:
        # Starting coordinates for the notebook layout (approx 52% width, 30% height)
        start_x = int(W * 0.52)
        y_cursor = int(H * 0.30)
        
        # Dark blue ink color for handwritten style
        INK_COLOR = (25, 35, 90, 255)
        
        # Determine fixed line height for stable spacing (measure first step or a typical string)
        test_bbox = FONT_HAND.getbbox("d = √((x₂-x₁)² + (y₂-y₁)²)gjy")
        line_height = (test_bbox[3] - test_bbox[1]) + 60  # includes padding
        
        # Jitter offsets to avoid rigid perfect alignment
        def get_jitter(text):
            return ((hash(text) % 15) - 7, (hash(text + "y") % 10) - 5)
            
        box_coords = None
        box_start_time = 0.0
        
        for step in active_steps:
            content_text = step["content"].strip()
            
            # Normalize Unicode subscripts to standard digits (Kalam font lacks them)
            # Using explicit unicode escapes to avoid Windows CP1252 source parsing bugs
            subscripts = str.maketrans("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089", "0123456789")
            content_text = content_text.translate(subscripts)
            
            # Sub-split into lines if there are any newlines
            lines = content_text.split("\n")
            
            # Record bounds of the final answer step to draw the box around it
            if step["category"] == "answer":
                step_y_start = y_cursor
                max_line_w = 0
                num_drawn_lines = 0
                max_len = 0
                for line in lines:
                    if line.strip():
                        bbox = FONT_HAND.getbbox(line)
                        line_w = bbox[2] - bbox[0]
                        if line_w > max_line_w:
                            max_line_w = line_w
                        num_drawn_lines += 1
                        if len(line) > max_len:
                            max_len = len(line)
                
                font_height = test_bbox[3] - test_bbox[1]
                x0_box = start_x - 18
                y0_box = step_y_start - 18
                x1_box = start_x + max_line_w + 18
                y1_box = step_y_start + (num_drawn_lines - 1) * line_height + font_height + 18
                box_coords = (x0_box, y0_box, x1_box, y1_box)
                box_start_time = step["start"] + (max_len / 3.0)

            for line in lines:
                if not line.strip():
                    y_cursor += line_height
                    continue
                
                # Calculate visible characters based on fixed 3.0 chars/sec speed
                chars_per_second = 3.0
                time_writing = max(0.0, t - step["start"])
                
                # Optional fade-in effect for the active character being typed
                visible_chars = min(len(line), int(time_writing * chars_per_second))
                visible_text = line[:visible_chars]
                
                jx, jy = get_jitter(line)
                draw.text(
                    (start_x + jx, y_cursor + jy),
                    visible_text,
                    font=FONT_HAND,
                    fill=INK_COLOR
                )
                
                # Advance cursor regardless of how much is currently visible
                # so that subsequent steps don't jump around.
                y_cursor += line_height
                
            # Extra padding between different logical steps
            y_cursor += 30
            
        # Draw hand-drawn box around final answer if visible and fully written
        if box_coords is not None and t > box_start_time:
            x0_box, y0_box, x1_box, y1_box = box_coords
            time_boxing = t - box_start_time
            overshoot = 6
            p1 = (x0_box - overshoot, y0_box)
            p2 = (x1_box + overshoot, y0_box)
            p3 = (x1_box, y0_box - overshoot)
            p4 = (x1_box, y1_box + overshoot)
            p5 = (x1_box + overshoot, y1_box)
            p6 = (x0_box - overshoot, y1_box)
            p7 = (x0_box, y1_box + overshoot)
            p8 = (x0_box, y0_box - overshoot)
            
            lines_to_draw = [
                (p1, p2, 0.0, 0.125, 101),   # Top Line
                (p3, p4, 0.125, 0.25, 202),  # Right Line
                (p5, p6, 0.25, 0.375, 303),  # Bottom Line
                (p7, p8, 0.375, 0.5, 404)    # Left Line
            ]
            
            for start_pt, end_pt, t_start, t_end, seed_val in lines_to_draw:
                if time_boxing >= t_end:
                    f = 1.0
                elif time_boxing <= t_start:
                    f = 0.0
                else:
                    f = (time_boxing - t_start) / (t_end - t_start)
                    
                if f > 0.0:
                    pts = get_wobbly_line(start_pt, end_pt, f, seed_val)
                    if len(pts) >= 2:
                        draw.line(pts, fill=INK_COLOR, width=3)

    return np.array(img.convert("RGB"))


video_clip = VideoClip(draw_frame, duration=duration)
video_clip = video_clip.set_audio(audio_clip)

# ── Write output ───────────────────────────────────────────────────────────────# Write video  (ffmpeg already on PATH from top of file)
output_path = OUTPUT_DIR / "output.mp4"
video_clip.write_videofile(
    str(output_path),
    fps=24,
    codec="libx264",
    audio_codec="aac",
    ffmpeg_params=["-pix_fmt", "yuv420p"]
)

print(f"\nDONE. Video generated: {output_path}")

# Probe output with ffmpeg binary used earlier to confirm pixel format
try:
    import subprocess
    probe = subprocess.run([_ffmpeg_exe, "-i", str(output_path)], capture_output=True)
    stderr = probe.stderr.decode(errors="replace")
    print("\n===== FFMPEG PROBE OUTPUT =====")
    print(stderr)
    print("===============================\n")
except Exception as e:
    print(f"Could not run ffmpeg probe: {e}")

# ── FINAL VERIFICATION ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PIPELINE COMPLETE - ARTIFACT VERIFICATION")
print("="*70)

print(f"\n[ARTIFACT PATHS]")
print(f"  * Transcript:  {transcript_json_path}")
print(f"  * Annotations: {annotations_json_path if annotations_raw_gemini else '(none - Gemini unavailable)'}")
print(f"  * Timeline:    {timeline_path}")
print(f"  * Video:       {output_path}")

print(f"\n[STATISTICS]")
print(f"  * Transcript segments: {len(segments)}")
print(f"  * Gemini annotations:  {len(annotations_raw_gemini) if annotations_raw_gemini else 0}")
print(f"  * Timeline events:     {len(annotations)}")
print(f"  * Video duration:      {duration:.2f} seconds")
print(f"  * Video resolution:    {W}x{H}")
print(f"  * Frames:             {int(duration * 24)}")

print("\n" + "="*70)
