# Page Reader

[![Support me on Patreon](https://img.shields.io/badge/Patreon-Support%20my%20work-FF424D?style=flat&logo=patreon&logoColor=white)](https://www.patreon.com/AndersBjarby)

A mobile-friendly web app that turns your phone's camera into an accessible reading aid. Point the camera at a page, capture it, and the app runs OCR on the text — then reads it aloud, translates it, simplifies it, or explains what's in the image.

## Features

- Camera capture with region selection
- OCR via Tesseract (OpenCV pre-processing)
- Read aloud with ElevenLabs text-to-speech (with audio caching)
- Translate text into many languages (Swedish, English, German, French, Arabic, Chinese, Japanese, and more)
- Simplify text and explain images using OpenAI

## Setup

Requires `OPENAI_API_KEY` and an ElevenLabs API key in a `.env` file, plus the Tesseract OCR engine installed on the system.

```bash
pip install flask numpy openai opencv-python pytesseract python-dotenv requests
python app.py
```

Runs on port 5000. Built to run on Replit.

## Tech

Flask, OpenCV, pytesseract, OpenAI, ElevenLabs TTS.
