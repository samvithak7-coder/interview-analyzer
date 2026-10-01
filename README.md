---
title: Interview Analyzer
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
short_description: AI-powered interview practice feedback, speech analytics, and STAR coaching rewrites.
---

# Interview Review & AI Delivery Coach

Upload an interview recording (audio or video), record live in the browser, or paste a transcript to receive instant AI scoring, STAR framework evaluation, delivery metrics (filler words, pace WPM), and coaching rewrites.

---

## 🚀 Live Demo & Hugging Face Spaces Deployment

This repository is configured for 1-click Docker deployment on **Hugging Face Spaces**.

### Deploying to Hugging Face Spaces
1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Select **Docker** as the Space SDK.
3. Push this repository to your Space:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push space main
   ```
4. Add your `GEMINI_API_KEY` in the Space settings under **Repository Secrets**.

---

## 🐳 Run locally with Docker

```bash
# Build the Docker image
docker build -t interview-analyzer .

# Run the container (pass your Gemini API key)
docker run -p 7860:7860 -e GEMINI_API_KEY="your_api_key_here" interview-analyzer
```
Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## 💻 Run locally without Docker

```bash
cd interview-analyzer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Add your GEMINI_API_KEY to .env
uvicorn app.main:app --reload
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## ⚙️ Features & Architecture

- **Speech-to-Text**: Powered by `faster-whisper` (`small` model) pre-compiled into the container.
- **AI Analytics**: Uses Google Gemini to split interview Q&A pairs, evaluate STAR responses, and generate practice plans.
- **Delivery Metrics**: Measures filler word counts, words per minute (WPM), and total duration.
- **Privacy First**: Audio/video recordings are processed in memory and immediately deleted after transcription.
