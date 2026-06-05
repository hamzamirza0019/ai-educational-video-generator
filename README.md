# Automatic Annotation Video Generator

This repository implements the end‑to‑end pipeline:

1. Convert `background.pdf` → `background.png` (300 dpi).
2. Transcribe `narration.wav` with **OpenAI Whisper** → `transcript.json` (includes timestamps).
3. Heuristic planner extracts annotation events from the transcript.
4. Pillow renderer draws highlights, equations, arrows, and text onto the background.
5. MoviePy assembles the rendered frames with the original audio → `output.mp4`.

## Usage
```bash
python main.py
```
The script expects the input files under `data/` (see folder layout).

## Folder Layout
```
task1/
├─ data/
│   ├─ background.pdf
│   └─ narration.wav
├─ background.png            # generated
├─ transcript.json           # generated
├─ annotation_timeline.json  # generated
├─ frames/                   # generated PNG sequence
└─ output.mp4                # final video
```
