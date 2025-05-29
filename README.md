# AURIS: Audio Understanding and Retrieval Inference System

AURIS is an intelligent audio processing system built using Python and Streamlit. It enables users to upload audio files or provide YouTube links to:

- Convert speech to text using offline speech recognition.
- Perform forced alignment using Gentle.
- Search and seek specific words within the audio playback.
- Interact with a dynamic, user-friendly interface.

---

## 🚀 Features

1. **Audio Upload**  
   Upload `.mp3` or `.wav` files directly.

2. **YouTube Audio Extraction**  
   Paste a YouTube link to extract the audio stream.

3. **Audio-to-Text Conversion**  
   Uses CMU Sphinx via the `SpeechRecognition` library for transcription.

4. **Forced Alignment**  
   Uses the [Gentle](https://github.com/lowerquality/gentle) server for word-level timestamp alignment.

5. **Word Search and Timestamp Seek**  
   Search for specific words and jump directly to their location in the audio.

6. **Clean Interface**  
   Built using Streamlit for a responsive, easy-to-use web experience.

---

## 🖥️ Run the Application

### 1. Install Requirements

```bash
pip install -r requirements.txt
