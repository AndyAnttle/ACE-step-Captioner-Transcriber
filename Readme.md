*[Read this in Russian (Русский)](https://github.com/AndyAnttle/ACE-step-Captioner-Transcriber/tree/russian-version)*

# 🎵 ACE-step: Captioner & Transcriber

![Application interface](screenshot.JPG)

An application for music captioning and song transcription using Qwen2.5-Omni ([ACE-step](https://huggingface.co/ACE-Step)) and Whisper models. It supports single files and folders of audio, noise reduction, stem separation (Demucs), and segmented processing.

## Features

- **Two recognition models**: Qwen Omni (for description/transcription) and Whisper Large v3/v2 (transcription only)
- **Flexible model loading**: quantization (4/8-bit), Flash Attention, offload to CPU
- **Batch processing**: multiple files or an entire folder
- **Audio preprocessing**:
  - Noise reduction (noisereduce)
  - Stem separation (Demucs) - extract vocals, instrumental, etc.
  - Trim by duration
  - Segmented processing for long audio
- **Automatic and manual saving** of results (next to the audio or into a chosen folder)
- **Convenient interface** built on Gradio with buttons to pick folders through the file explorer
- **GPU support** for acceleration (Demucs, models)

## Installation

### Requirements
- Python 3.12 (recommended) or 3.10-3.12. Tested on Python 3.12.9
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
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows
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

The browser will open with the interface after launch.

### Usage
1. Select the recognition model (Qwen Omni or Whisper).

2. If Qwen is chosen, set the path to the folder with models (you can change it in the code: MODELS_ROOT = r"your\path\folder") and load the model.

3. For Whisper, choose the version and language, then load the model.

4. Adjust preprocessing settings (noise reduction, stems, segmentation, etc.).

5. Choose the input mode: single file, multiple files, or a folder.

6. Click the run button (its label depends on the selected model).

7. The result will appear in the text box. You can save it manually or enable auto-save.

### Notes

Qwen Omni requires a specific transformers build from a branch; it is pinned in requirements.txt.

On first launch Demucs will download its model weights (~1 GB) into the cache.

Whisper large-v3 is downloaded on first use (~3 GB).
