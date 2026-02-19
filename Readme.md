*[Read this in Russian (Русский)](https://github.com/AndyAnttle/ACE-step-Captioner-Transcriber/tree/russian-version)*

# 🎵 ACE-step: Captioner & Transcriber

![Image](screenshot.JPG)

A powerful application for music captioning and song transcription using state-of-the-art models: **Qwen2.5-Omni ([ACE-Step](https://huggingface.co/ACE-Step))** and **Whisper** (large-v3 / large-v2). It offers flexible preprocessing (noise reduction, stem separation, segmentation) and batch processing.

## Features

- **Two recognition models**: Qwen Omni (for music description / transcription) and Whisper (pure transcription).
- **Flexible model loading**: quantization (4/8-bit), Flash Attention, CPU offload.
- **Batch processing**: process multiple files or an entire folder.
- **Audio preprocessing**:
  - Noise reduction (noisereduce)
  - Stem separation (Demucs) – extract vocals, instrumental, etc.
  - Trim by duration
  - Segmented processing for long audio
- **Auto-save and manual save** results (next to the audio or in a specified folder).
- **User-friendly Gradio interface** with native folder selection dialogs.
- **GPU acceleration** for Demucs and models.

## Installation

### Requirements
- Python 3.12 (recommended) or 3.10–3.12.
	Tested on Python 3.12.9
- CUDA (optional, for GPU)

### Steps

1. Clone the repository:
```
   git clone https://github.com/AndyAnttle/ACE-step-Captioner-Transcriber.git
   cd ACE-step-Captioner-Transcriber
```

2. Create a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate       # Windows
```

3. Install dependencies:
Torch and requirements for Python 3.12.9
```
pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

4. Run the application:
```
python app.py
```
The browser will open automatically with the interface.

### Usage
1. Select the recognition model (Qwen Omni or Whisper).

2. If Qwen is chosen, specify the path to your Qwen models; (you can change it in the code MODELS_ROOT = r"your\path\folder") and load the model.

3. For Whisper, choose the version and language, then load the model.

4. Adjust preprocessing settings (noise reduction, stems, segmentation, etc.).

5. Choose input mode: single file, multiple files, or a folder.

6. Click the run button (its label depends on the selected model).

7. The result will appear in the text box. You can save it manually or enable auto-save.


### Notes
Qwen Omni requires a specific transformers branch; it is specified in requirements.txt.

Demucs will download its weights (~1 GB) on first use.


Whisper large-v3 will be downloaded on first use (~3 GB).





