# AI Educational Video Generator

An automated educational video generation pipeline that creates synchronized annotations from a question PDF/image and teacher narration audio.

## Architecture

PDF/Image
↓ OCR (EasyOCR)
ocr.json

Audio
↓ Faster-Whisper
transcript.json

OCR + Transcript
↓ Gemini
annotations.json

Annotations
↓ Timeline Builder
annotation_timeline.json

MoviePy + Pillow
↓
output.mp4

## Features

* OCR extraction from educational content
* Speech transcription with timestamps
* AI-generated educational annotations
* Timeline synchronization
* Automatic video rendering
* JSON artifact generation for debugging and inspection

## Tech Stack

* Python
* EasyOCR
* Faster-Whisper
* Google Gemini
* Pillow
* MoviePy
* FFmpeg

## Setup

```bash
pip install -r requirements.txt
```

Set:

```bash
GOOGLE_API_KEY=your_key
```

## Run

```bash
python run_pipeline.py
```

## Output Artifacts

Generated inside `output/`:

* ocr.json
* transcript.json
* annotations.json
* annotation_timeline.json
* output.mp4

## Limitations

* Annotation quality depends on OCR quality.
* Synchronization depends on transcript segmentation.
* Complex diagrams may require additional planning logic.
