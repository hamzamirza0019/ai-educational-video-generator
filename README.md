# AI Educational Video Generator

An automated pipeline that converts a question PDF/image and teacher narration audio into a synchronized educational video with handwritten-style solution annotations.

The system combines OCR, speech transcription, AI-driven annotation planning, timeline synchronization, and video rendering to generate teacher-style explanation videos automatically.

---

# Features

* Extracts educational content from PDFs/images using OCR
* Transcribes teacher narration with timestamps
* Generates solution annotations using Gemini
* Synchronizes annotations with spoken explanations
* Renders handwritten-style solutions directly on the question sheet
* Progressive writing animation synchronized with narration
* Automatic answer highlighting
* Produces intermediate JSON artifacts for debugging and inspection

---

# System Architecture

```text
Question PDF/Image
        │
        ▼
     EasyOCR
        │
        ▼
     ocr.json

Teacher Narration
        │
        ▼
 Faster-Whisper
        │
        ▼
 transcript.json

 OCR + Transcript
        │
        ▼
 Gemini Annotation Planner
        │
        ▼
 annotations.json

 Timeline Generator
        │
        ▼
 annotation_timeline.json

 MoviePy + Pillow Renderer
        │
        ▼
     output.mp4
```

---

# Example Workflow

Input:

* Question PDF/Image
* Teacher narration audio

Output:

* Educational video containing:

  * Original question
  * Teacher-style handwritten solution
  * Synchronized annotation appearance
  * Final answer highlighting

---

# Project Structure

```text
task1/
│
├── data/
│   ├── background.pdf
│   ├── background.png
│   └── narration.wav
│
├── output/
│   ├── ocr.json
│   ├── transcript.json
│   ├── annotations.json
│   ├── annotation_timeline.json
│   └── output.mp4
│
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

# Tech Stack

### OCR

* EasyOCR

### Speech Recognition

* Faster-Whisper

### AI Annotation Planning

* Google Gemini

### Rendering

* Pillow
* MoviePy
* FFmpeg

### Language

* Python

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd task1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Setup

Create a Gemini API key and set:

```bash
GOOGLE_API_KEY=your_api_key
```

Windows PowerShell:

```powershell
$env:GOOGLE_API_KEY="your_api_key"
```

---

# Running the Pipeline

Place:

* Question PDF/Image inside `data/`
* Narration audio inside `data/`

Run:

```bash
python run_pipeline.py
```

---

# Generated Outputs

The pipeline generates the following artifacts:

| File                     | Description                           |
| ------------------------ | ------------------------------------- |
| ocr.json                 | OCR extracted text and bounding boxes |
| transcript.json          | Whisper transcription with timestamps |
| annotations.json         | Gemini-generated annotation plan      |
| annotation_timeline.json | Time-aligned annotation schedule      |
| output.mp4               | Final educational video               |

---

# Annotation Categories

The system currently supports:

* Question emphasis
* Given values
* Formula display
* Substitution steps
* Calculation steps
* Final answer rendering
* Correct option highlighting

---

# Design Goals

The project aims to:

* Reduce manual educational video production effort
* Synchronize visual explanations with spoken narration
* Generate teacher-style handwritten solution videos
* Provide a modular pipeline where each stage can be inspected independently

---

# Limitations

* Annotation quality depends on OCR quality.
* Transcription quality depends on narration clarity.
* Complex diagrams may require additional reasoning and layout planning.
* AI-generated annotations depend on Gemini output quality.
* The current MVP is optimized for educational question-solving workflows.

---

# Future Improvements

* Multi-page PDF support
* Diagram-aware annotations
* Subject-specific planners
* Better mathematical notation rendering
* Dynamic layout planning
* Real-time video generation

---

# License

This project was developed as part of an AI educational video generation internship task and is intended for learning, experimentation, and research purposes.
