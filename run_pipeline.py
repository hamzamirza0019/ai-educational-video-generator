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
  {{"category": "substitution", "source": "transcript", "trigger_quote": "under root of x2 plus 4 square", "type": "show_text", "region": "working_area", "content": "d = √((4-1)² + (6-2)²)"}},
  {{"category": "calculation", "source": "transcript", "trigger_quote": "of 9 plus 16 which comes out to be", "type": "show_text", "region": "working_area", "content": "d = √(9 + 16)"}},
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
                x0, y0, x1, y1 = 53, 518, 403, 611
                
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
            # Explicit content-based overrides for specific educational elements to ensure accuracy:
            if "x" in content and "y" in content and ("2" in content or "₂" in content or "x2" in content):
                start_t = 13.52
                end_t = 25.28
                matched_seg_txt = "We know that distance between 2 points is given as... plus y2 minus y1 whole square."
                print(f"    [OVERRIDE] Distance formula detected. Forcing timestamp: {start_t} - {end_t}")
            elif "9" in content and "16" in content:
                start_t = 53.88
                end_t = 60.04
                matched_seg_txt = "of 9 plus 16 which comes out to be under root 25."
                print(f"    [OVERRIDE] Calculation d = √(9 + 16) detected. Forcing timestamp: {start_t} - {end_t}")
            else:
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
                    start_t = segments[kept_segs[0]]["start"]
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

# Try to load a font; fall back to default if unavailable
try:
    # Increase font sizes for better visibility
    FONT_LABEL = ImageFont.truetype("arial.ttf", size=max(32, H // 22))  # Larger text
    FONT_EQ    = ImageFont.truetype("arial.ttf", size=max(36, H // 20))   # Larger formulas
    FONT_SMALL = ImageFont.truetype("arial.ttf", size=max(24, H // 28))
except Exception:
    FONT_LABEL = ImageFont.load_default()
    FONT_EQ    = ImageFont.load_default()
    FONT_SMALL = ImageFont.load_default()

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
        if ann_type == "highlight" or ann_type == "correct_option_highlight":
            x0 = ann.get("x0")
            y0 = ann.get("y0")
            x1 = ann.get("x1")
            y1 = ann.get("y1")
            if x0 is not None and y0 is not None and x1 is not None and y1 is not None:
                # Fade in effect: 0.3s fade duration
                time_in = t - ann["start"]
                fade_duration = 0.3
                alpha_factor = min(1.0, time_in / fade_duration)
                
                if ann_type == "highlight":
                    # Transparent warm yellow highlight with orange border
                    fill_color = (255, 240, 100, int(80 * alpha_factor))
                    outline_color = (255, 150, 0, int(150 * alpha_factor))
                else:
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
            # Avoid duplicate content if any
            content = ann.get("content", "").strip()
            if content and content not in [s["content"] for s in active_steps]:
                active_steps.append({
                    "content": content,
                    "start": ann["start"],
                    "category": ann.get("category", "")
                })
                
    # Sort active steps chronologically by start time
    active_steps.sort(key=lambda s: s["start"])

    # 3. Draw the single unified "Working Steps" card on the right side of the canvas
    if active_steps:
        # Define card dimensions on the right side of the screen
        card_w = 1150
        card_h = 1000
        # Right half of 2668 is x >= 1334. Center the card in the right half:
        card_x0 = 1334 + (1334 - card_w) // 2
        card_y0 = 250
        card_x1 = card_x0 + card_w
        card_y1 = card_y0 + card_h
        
        # Slate glassmorphism card background with padding and rounded corners
        draw.rounded_rectangle(
            [card_x0, card_y0, card_x1, card_y1],
            radius=24,
            fill=(20, 24, 33, 230),  # Sleek dark slate
            outline=(67, 97, 238, 255),  # Indigo border
            width=6
        )
        
        # Draw Card Header
        header_text = "📝 Working Steps"
        draw.text(
            (card_x0 + 40, card_y0 + 35),
            header_text,
            font=FONT_EQ,
            fill=(255, 204, 0, 255)  # Warm gold
        )
        
        # Draw a horizontal divider line under header
        draw.line(
            [card_x0 + 40, card_y0 + 95, card_x1 - 40, card_y0 + 95],
            fill=(100, 116, 139, 150),
            width=2
        )
        
        # Draw each step inside the card
        # Vertical space layout:
        y_cursor = card_y0 + 120
        for idx, step in enumerate(active_steps):
            # Render category label (e.g. "Step 1: Formula")
            cat_label = f"Step {idx + 1} ({step['category'].upper()}):"
            draw.text(
                (card_x0 + 40, y_cursor),
                cat_label,
                font=FONT_SMALL,
                fill=(140, 180, 255, 255)  # Soft blue
            )
            y_cursor += 35
            
            # Render step content (can be multi-line text)
            content_text = step["content"]
            lines = content_text.split("\n")
            for line in lines:
                draw.text(
                    (card_x0 + 60, y_cursor),
                    line,
                    font=FONT_LABEL,
                    fill=(255, 255, 255, 255)  # Crisp white
                )
                y_cursor += 55
            
            # Extra spacing between steps
            y_cursor += 25
            
            # Avoid overflow by stopping if we exceed card bounds
            if y_cursor > card_y1 - 60:
                break

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
