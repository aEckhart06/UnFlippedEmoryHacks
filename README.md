# UnFlipped
**AI-Powered Interactive Lecture Platform** · 🏆 Winner, Emory Hacks 2025

---

## Introduction / Executive Summary

### The Problem
In a "flipped classroom" model, students watch lecture videos on their own time and work through problems in class. It's an effective model on paper, but it breaks down in one specific way: **when a student is confused at 2am watching a recording, there's no one to ask.** Office hours are asynchronous, forums are slow, and generic chatbots don't know what the student has actually watched or where they are in the material.

That leaves three unmet requirements for any real solution:
- **Active engagement**, not passive video playback
- **Comprehension checks** that are grounded in the specific content a student has seen
- **A single, unified tool** — not five disconnected apps stitched together

### The Solution
**UnFlipped** turns a static lecture video into an interactive tutoring session. A student uploads a recording, and the system automatically transcribes it, tracks where the student is in the video, and makes that moment queryable — the student can ask a question and get an answer from an AI "professor" that's grounded in exactly what's been covered so far. Periodically, the system also generates short multiple-choice checkpoints from the material already watched, so students get comprehension feedback without waiting for a quiz or exam.

The result is a self-contained web application that takes a video from **upload → transcription → interactive Q&A → comprehension check**, with no manual setup between steps.

---

## Architecture

### Diagram

[System Architecture Diagram](unflipped_architecture_diagram.webp)
🚧 = in progress — see [Unfinished Features](#unfinished-features) below.

### Architecture Breakdown & Design Choices

| Layer | Choice | Why |
|---|---|---|
| **Web framework** | Flask | Fast to stand up a full-stack app with server-rendered templates for a hackathon timeline |
| **Storage** | MongoDB + GridFS | One datastore for both video binaries (GridFS) and transcript/metadata documents — avoids standing up a separate object store for a prototype |
| **Speech-to-text** | OpenAI Whisper (`tiny`, local inference) | Chose the smallest model to keep transcription fast on CPU-only hardware, trading a few points of word-error-rate for roughly 2x the speed |
| **LLM routing** | GPT-4o for Q&A, GPT-3.5-turbo for MCQ generation | Tiered by task: reasoning-heavy, timestamp-aware tutoring goes to the stronger model; cheap structured JSON generation for quizzes goes to the faster/cheaper one |
| **Async processing** | Background thread per upload, with client-side polling | Keeps the upload endpoint responsive while Whisper transcribes in the background, instead of blocking the request |
| **Video processing** | OpenCV for first-frame thumbnail extraction | Avoids re-encoding video just to generate a library preview image |

**Data flow:** upload → GridFS blob storage → background transcription → timestamped transcript stored in MongoDB → client queries against the current video timestamp → LLM response rendered back into the UI.

### Key Components & Quantitative Results

| Component | Status | Notes |
|---|---|---|
| Video ingestion (GridFS) | ✅ Implemented | Unified blob storage alongside transcript metadata |
| Background transcription pipeline | ✅ Implemented | Threaded worker + status polling endpoint |
| Timestamp-aware Q&A (GPT-4o) | ✅ Implemented | Full transcript currently sent per query — see roadmap |
| MCQ generation (GPT-3.5-turbo) | ✅ Implemented | Transcript segmented by time window before generation |
| Thumbnail extraction (OpenCV) | ✅ Implemented | Cached via `Cache-Control` headers |
| Timestamp-scoped retrieval (RAG) | 🚧 In progress | Will replace full-transcript prompting with a segment-window filter |
| Benchmarking suite (latency, WER, token cost) | 🚧 In progress | No quantitative metrics published yet — see roadmap |

---

## Conclusion

### Next Steps
1. **Timestamp-scoped context retrieval** — filter transcript segments to the student's current playhead window instead of sending the full transcript, cutting both token cost and latency.
2. **Quantitative benchmarking** — publish transcription latency, Whisper WER, GPT-4o response latency, and token cost per query.
3. **One-command local setup** (Docker Compose) — package Flask, MongoDB, and any background workers so the app runs with a single command.
4. **Recover the background job queue** (Celery + Redis) — replace the current thread-per-upload approach with a proper worker queue for reliability under concurrent load.
5. **Wire up LLM output validation** — enforce structured JSON output on MCQ generation with server-side validation and fallback handling.

### Unfinished Features
The following are scoped and partially started, but not yet complete:
- 🚧 **Timestamp-scoped retrieval (RAG)** — currently the full transcript is sent on every query rather than a relevant window
- 🚧 **Authentication** — login/signup UI exists, but backend verification is not yet wired up
- 🚧 **Task queue (Celery + Redis)** — background transcription currently runs on a simple thread; a queue-based version is in progress
- 🚧 **Automated testing & CI** — no test suite or CI pipeline yet
- 🚧 **Containerized setup** — no Docker Compose configuration yet; local setup is manual
- 🚧 **Structured logging & job status tracking** — currently relies on basic console output

---

*UnFlipped was built by Drew, Surya, and Aryan at Emory Hacks 2025.*
