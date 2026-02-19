import os
import time
import gc
import re
import threading
import numpy as np
import gradio as gr
import torch
import librosa
import soundfile as sf
import tempfile
from pathlib import Path
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    TextStreamer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline
)
from qwen_omni_utils import process_mm_info
import tkinter as tk
from tkinter import filedialog
import noisereduce as nr
import torchaudio
from demucs import pretrained
from demucs.apply import apply_model

# ----------------------------------------------------------------------
# Stop criterion for interrupting generation (Qwen only)
# ----------------------------------------------------------------------
class StopOnEvent(StoppingCriteria):
    def __init__(self, stop_event):
        self.stop_event = stop_event
    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event.is_set()

# ----------------------------------------------------------------------
# Configuration and global variables
# ----------------------------------------------------------------------
MODELS_ROOT = r"your\path\folder"
SAMPLE_RATE = 16000
DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

current_model = None           # for Qwen
current_processor = None       # for Qwen
whisper_model = None           # for Whisper
whisper_processor = None       # for Whisper
whisper_pipeline = None        # Whisper pipeline
current_model_path = None
current_model_type = None
current_stop_event = None
batch_stop_flag = False

# ----------------------------------------------------------------------
# Qwen model search (simple and advanced)
# ----------------------------------------------------------------------
def find_models_simple(root_dir):
    models = []
    if not os.path.exists(root_dir):
        return models
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            lower = item.lower()
            if "captioner" in lower:
                models.append({"name": item, "path": item_path, "type": "captioner"})
            elif "transcriber" in lower:
                models.append({"name": item, "path": item_path, "type": "transcriber"})
    return models

def find_models_advanced(root_dir):
    models = []
    if not os.path.exists(root_dir):
        return models
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
            lower = item.lower()
            if "captioner" in lower:
                model_type = "captioner"
            elif "transcriber" in lower:
                model_type = "transcriber"
            else:
                model_type = "captioner"  # default
            models.append({"name": item, "path": item_path, "type": model_type})
    return models

def get_models(advanced):
    if advanced:
        return find_models_advanced(MODELS_ROOT)
    else:
        return find_models_simple(MODELS_ROOT)

# ----------------------------------------------------------------------
# Unload Qwen
# ----------------------------------------------------------------------
def unload_qwen_model():
    global current_model, current_processor, current_model_path, current_model_type
    if current_model is not None:
        print("\n=== UNLOADING QWEN ===")
        del current_model
        del current_processor
        current_model = None
        current_processor = None
        current_model_path = None
        current_model_type = None
        gc.collect()
        torch.cuda.empty_cache()
        print("Qwen unloaded.")

# ----------------------------------------------------------------------
# Load Qwen
# ----------------------------------------------------------------------
def load_qwen_model(model_info, dtype_choice, quantization_choice, use_flash, disable_talker,
                    device_map_strategy, custom_limits_str, offload_folder):
    global current_model, current_processor, current_model_path, current_model_type

    unload_qwen_model()

    path = model_info["path"]
    model_type = model_info["type"]

    if quantization_choice != "None":
        try:
            import bitsandbytes
        except ImportError:
            return (gr.update(), "❌ bitsandbytes not installed, quantization unavailable", gr.update())

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float8_e4m3fn": torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else None,
        "float8_e5m2": torch.float8_e5m2 if hasattr(torch, 'float8_e5m2') else None
    }
    torch_dtype = dtype_map.get(dtype_choice, torch.float16)
    if torch_dtype is None:
        return (gr.update(), f"❌ Type {dtype_choice} not supported by this torch version", gr.update())

    quantization_config = None
    if quantization_choice == "8-bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype = torch.float16
    elif quantization_choice == "4-bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        torch_dtype = torch.float16

    device_map = device_map_strategy if device_map_strategy != "custom" else "auto"
    max_memory = None
    if device_map_strategy == "custom":
        max_memory, err = parse_memory_limits(custom_limits_str)
        if err:
            return (gr.update(), f"❌ Error in limits: {err}", gr.update())

    attn_impl = "flash_attention_2" if use_flash else None

    print(f"\n=== LOADING QWEN: {model_info['name']} ===")
    start_load = time.time()
    try:
        processor = Qwen2_5OmniProcessor.from_pretrained(path, trust_remote_code=True)
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            offload_folder=offload_folder if offload_folder else None
        )
        if disable_talker and hasattr(model, "disable_talker"):
            model.disable_talker()
            print("Talker disabled.")
        model.eval()
    except Exception as e:
        print(f"ERROR: {e}")
        return (gr.update(), f"❌ Failed to load model: {e}", gr.update())

    current_model = model
    current_processor = processor
    current_model_path = path
    current_model_type = model_type

    load_time = time.time() - start_load
    print(f"Qwen loaded in {load_time:.2f} seconds")
    print(f"Devices: {model.device}, dtype: {model.dtype}")

    if model_type == "captioner":
        default_prompt = "*Task* Describe this audio in detail. Provide only textual description."
        btn_text = "🎵 Describe music"
    else:
        default_prompt = "*Task* Transcribe this audio in detail. Provide structured lyrics with section tags."
        btn_text = "📝 Transcribe"

    btn_update = gr.update(value=btn_text, variant="primary", interactive=True)
    status_msg = f"✅ Model {model_info['name']} loaded (time: {load_time:.1f}s)"
    prompt_update = gr.update(value=default_prompt)

    return (btn_update, status_msg, prompt_update)

# ----------------------------------------------------------------------
# Load Whisper (model and pipeline)
# ----------------------------------------------------------------------
def load_whisper_model(model_version="openai/whisper-large-v3"):
    global whisper_model, whisper_processor, whisper_pipeline
    if whisper_pipeline is not None and hasattr(whisper_model, "config") and whisper_model.config._name_or_path == model_version:
        return
    unload_whisper_model()
    device = 0 if torch.cuda.is_available() else -1
    print(f"📥 Loading Whisper {model_version}...")
    whisper_processor = WhisperProcessor.from_pretrained(model_version)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(model_version).to(device if device >= 0 else "cpu")
    whisper_model.eval()
    whisper_pipeline = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        device=device,
    )
    print(f"✅ Whisper {model_version} loaded")

# ----------------------------------------------------------------------
# Unload Whisper
# ----------------------------------------------------------------------
def unload_whisper_model():
    global whisper_model, whisper_processor, whisper_pipeline
    if whisper_pipeline is not None:
        print("\n=== UNLOADING WHISPER ===")
        del whisper_pipeline
        del whisper_model
        del whisper_processor
        whisper_pipeline = None
        whisper_model = None
        whisper_processor = None
        gc.collect()
        torch.cuda.empty_cache()
        print("Whisper unloaded.")

# ----------------------------------------------------------------------
# Parse memory limits
# ----------------------------------------------------------------------
def parse_memory_limits(limit_str):
    if not limit_str or limit_str.strip() == "":
        return None, None
    parts = [p.strip() for p in limit_str.split(",")]
    max_memory = {}
    pattern = re.compile(r"^(\d+(?:\.\d+)?)\s*(GB|MB|GiB|MiB)?$", re.IGNORECASE)
    for i, part in enumerate(parts):
        if not part:
            continue
        m = pattern.match(part)
        if not m:
            return None, f"Invalid format for GPU {i}: '{part}'"
        num = float(m.group(1))
        unit = (m.group(2) or "GB").upper()
        if unit in ("GB", "GIB"):
            pass
        elif unit in ("MB", "MIB"):
            num = num / 1024.0
        else:
            return None, f"Unknown unit '{unit}'"
        max_memory[i] = f"{num:.0f}GB"
    return max_memory, None

# ----------------------------------------------------------------------
# Helper functions for saving results
# ----------------------------------------------------------------------
def ensure_output_dir(output_dir):
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

def save_result(text, audio_path, output_dir, model_type):
    if not text or text.startswith("⚠") or text.startswith("Error") or text.startswith("⏹️"):
        return None, "Result not saved (empty or erroneous)"
    if audio_path and isinstance(audio_path, str) and os.path.exists(audio_path):
        base = os.path.splitext(os.path.basename(audio_path))[0]
    else:
        base = "result"
    suffix = "caption" if model_type == "captioner" else "transcript"
    filename = f"{base}_{suffix}.txt"
    if output_dir:
        ensure_output_dir(output_dir)
        save_path = os.path.join(output_dir, filename)
    elif audio_path and isinstance(audio_path, str):
        save_path = os.path.join(os.path.dirname(audio_path), filename)
    else:
        save_path = filename
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return save_path, f"✅ Result saved: {save_path}"
    except Exception as e:
        return None, f"❌ Save error: {e}"

def save_debug_audio(audio_data, sr, prefix="qwen_input"):
    """Save audio to temporary folder for debugging."""
    timestamp = int(time.time())
    filename = f"{prefix}_{timestamp}.wav"
    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, filename)
    sf.write(path, audio_data, sr)
    print(f"💾 DEBUG: audio saved to {path}")
    return path

# ----------------------------------------------------------------------
# Folder selection dialog
# ----------------------------------------------------------------------
def browse_folder(current_path=""):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    initial_dir = current_path if current_path and os.path.exists(current_path) else "."
    selected = filedialog.askdirectory(title="Select folder", initialdir=initial_dir)
    root.destroy()
    return selected if selected else current_path

# ----------------------------------------------------------------------
# Stem separation (Demucs)
# ----------------------------------------------------------------------
def separate_stems(audio_path, target_stem='vocals', use_gpu=False):
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"🎤 Loading Demucs model (htdemucs) on {device}...")
    model = pretrained.get_model('htdemucs')
    model.to(device)
    model.eval()

    wav, orig_sr = torchaudio.load(audio_path)
    target_sr = 44100
    if orig_sr != target_sr:
        print(f"🔁 Resampling from {orig_sr} Hz to {target_sr} Hz...")
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        wav = resampler(wav)
        sr = target_sr
    else:
        sr = orig_sr

    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    wav = wav.unsqueeze(0).to(device)

    print("🔄 Separating stems...")
    with torch.no_grad():
        sources = apply_model(model, wav, shifts=5, split=True, overlap=0.25, progress=True)[0]

    stems = model.sources  # ['drums', 'bass', 'other', 'vocals']

    if target_stem == 'instrumental':
        vocal_idx = stems.index('vocals')
        instrumental = torch.cat([sources[i] for i in range(len(stems)) if i != vocal_idx], dim=0).mean(dim=0, keepdim=True)
        result_audio = instrumental.cpu().numpy().squeeze()
    else:
        stem_idx = stems.index(target_stem)
        result_audio = sources[stem_idx].cpu().numpy().squeeze()
        if result_audio.ndim > 1:
            result_audio = result_audio.mean(axis=0)

    # Resample back to 16 kHz for the models
    if sr != SAMPLE_RATE:
        print(f"🔁 Resampling result from {sr} Hz to {SAMPLE_RATE} Hz...")
        result_audio = librosa.resample(result_audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    return result_audio, SAMPLE_RATE

# ----------------------------------------------------------------------
# Split audio into segments
# ----------------------------------------------------------------------
def split_audio(audio_data, sr, segment_sec, overlap_sec=0):
    samples_per_seg = int(segment_sec * sr)
    total_samples = len(audio_data)
    segments = []
    start = 0
    while start < total_samples:
        end = min(start + samples_per_seg, total_samples)
        segment = audio_data[start:end]
        segments.append(segment)
        start += samples_per_seg
    return segments

# ----------------------------------------------------------------------
# Whisper transcription (pipeline, sequential)
# ----------------------------------------------------------------------
def transcribe_with_whisper_pipeline(audio_data, sr, language=None):
    global whisper_pipeline
    if whisper_pipeline is None:
        load_whisper_model()
    if sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000
    generate_kwargs = {
        "task": "transcribe",
        "temperature": 0.0,
    }
    if language is not None and language != "auto":
        generate_kwargs["language"] = language
    result = whisper_pipeline(audio_data, generate_kwargs=generate_kwargs)
    return result["text"]

# ----------------------------------------------------------------------
# Whisper transcription (manual segmentation)
# ----------------------------------------------------------------------
def transcribe_with_whisper_manual(audio_data, sr, language=None):
    global whisper_model, whisper_processor
    if whisper_model is None:
        load_whisper_model()
    if sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000
    inputs = whisper_processor(audio_data, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(whisper_model.device) for k, v in inputs.items()}
    forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language=language, task="transcribe")
    with torch.no_grad():
        predicted_ids = whisper_model.generate(
            **inputs,
            forced_decoder_ids=forced_decoder_ids,
        )
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# ----------------------------------------------------------------------
# Qwen processing
# ----------------------------------------------------------------------
def process_with_qwen(audio_data, sr, text_input, system_prompt, user_prompt,
                      max_new_tokens, do_sample, temperature, repetition_penalty,
                      console_progress):
    global current_model, current_processor, current_model_type, current_stop_event

    if current_model is None:
        return "⚠️ Load a Qwen model first."

    # Build conversation
    conversations = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]
    user_content = [{"type": "audio", "audio": audio_data}, {"type": "text", "text": user_prompt}]
    conversations.append({"role": "user", "content": user_content})

    # Tokenize
    try:
        audios_list = [audio_data]
        inputs = current_processor.apply_chat_template(
            conversations,
            audios=audios_list,
            load_audio_from_video=False,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(current_model.device)
    except Exception as e:
        return f"Tokenization error: {e}"

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v) and v.dtype != current_model.dtype:
            inputs[k] = v.to(current_model.dtype)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": current_processor.tokenizer.pad_token_id or current_processor.tokenizer.eos_token_id,
        "do_sample": do_sample,
        "return_audio": False,
        "repetition_penalty": repetition_penalty,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    if console_progress:
        streamer = TextStreamer(current_processor.tokenizer, skip_prompt=True)
        gen_kwargs["streamer"] = streamer

    stop_event = threading.Event()
    current_stop_event = stop_event
    stopping_criteria = StoppingCriteriaList([StopOnEvent(stop_event)])
    gen_kwargs["stopping_criteria"] = stopping_criteria

    try:
        with torch.no_grad():
            output = current_model.generate(**inputs, **gen_kwargs)
    except Exception as e:
        return f"Generation error: {e}"
    finally:
        current_stop_event = None

    # Decode
    if isinstance(output, torch.Tensor):
        text_ids = output
    elif isinstance(output, tuple):
        text_ids = output[0]
    else:
        text_ids = output

    input_len = inputs["input_ids"].shape[1]
    response_ids = text_ids[0][input_len:]
    response = current_processor.decode(response_ids, skip_special_tokens=True)

    if stop_event.is_set():
        response = "⚠️ Generation interrupted by user (partial result).\n\n" + response

    return response

# ----------------------------------------------------------------------
# Single file processing (unified preprocessing + model call)
# ----------------------------------------------------------------------
def process_single_file(audio_file, text_input, system_prompt, user_prompt,
                        max_sec, max_new_tokens, do_sample, temperature, repetition_penalty,
                        console_progress, noise_reduction, use_stem_separation, target_stem,
                        use_gpu_demucs, save_debug, recognition_model, use_segmentation,
                        segment_duration, whisper_language):
    """
    Process a single audio file:
    - Preprocessing (stems, noise reduction, trim, normalization)
    - Then pass to the appropriate model with segmentation handling
    """
    global batch_stop_flag

    # Handle tuple input from Gradio
    if isinstance(audio_file, tuple) and len(audio_file) == 2:
        sr, arr = audio_file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, arr, sr)
        audio_file = tmp.name

    if not os.path.exists(audio_file):
        return "Error: audio file not found."

    try:
        if use_stem_separation:
            print(f"🎤 Applying stem separation (target: {target_stem})...")
            try:
                audio_data, sr = separate_stems(audio_file, target_stem, use_gpu=use_gpu_demucs)
            except Exception as e:
                print(f"❌ Separation error: {e}. Using original audio.")
                audio_data, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
        else:
            audio_data, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)

        print(f"    Sample rate: {sr} Hz, shape: {audio_data.shape}")
        dur = len(audio_data) / sr
        print(f"    Duration: {dur:.2f} s")

        if noise_reduction:
            print("    Applying noise reduction...")
            audio_data = nr.reduce_noise(y=audio_data, sr=sr, prop_decrease=0.8)

        if max_sec > 0:
            max_samp = int(max_sec * sr)
            if len(audio_data) > max_samp:
                audio_data = audio_data[:max_samp]
                print(f"    Trimmed to {max_sec} s")

        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            print("    Normalized")

        if save_debug:
            save_func = globals()['save_debug_audio']
            save_func(audio_data, sr, prefix="processed_audio")

    except Exception as e:
        return f"Audio loading error: {e}"

    # Model selection
    if recognition_model == "Whisper Large v3":
        if use_segmentation:
            # Manual segmentation for Whisper
            segments = split_audio(audio_data, sr, segment_duration)
            print(f"Manual segmentation: {len(segments)} segments")
            full_text = ""
            for idx, seg in enumerate(segments):
                if batch_stop_flag:
                    full_text += "\n⏹️ Processing interrupted"
                    break
                print(f"--- Segment {idx+1}/{len(segments)} ---")
                seg_text = transcribe_with_whisper_manual(seg, sr, language=whisper_language if whisper_language != "auto" else None)
                full_text += seg_text + " "
            return full_text.strip()
        else:
            # Sequential processing via pipeline
            return transcribe_with_whisper_pipeline(audio_data, sr, language=whisper_language if whisper_language != "auto" else None)
    else:
        # Qwen Omni
        return process_with_qwen(audio_data, sr, text_input, system_prompt, user_prompt,
                                 max_new_tokens, do_sample, temperature, repetition_penalty,
                                 console_progress)

# ----------------------------------------------------------------------
# Batch processing (multiple files or folder)
# ----------------------------------------------------------------------
def batch_process(file_list, folder_path, text_input, system_prompt, user_prompt,
                  max_audio_sec, max_new_tokens, do_sample, temperature,
                  repetition_penalty, auto_save, output_dir, console_progress,
                  noise_reduction, use_segmentation, segment_duration,
                  use_stem_separation, target_stem, use_gpu_demucs, save_debug,
                  recognition_model, whisper_language, progress=gr.Progress()):
    global batch_stop_flag, current_model_type

    if recognition_model == "Qwen Omni" and current_model is None:
        return "⚠️ Load a Qwen model first."
    if recognition_model == "Whisper Large v3":
        load_whisper_model()

    audio_files = []
    if file_list:
        audio_files.extend(file_list)
    if folder_path and os.path.isdir(folder_path):
        for f in os.listdir(folder_path):
            if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
                audio_files.append(os.path.join(folder_path, f))

    if not audio_files:
        return "No audio files to process."

    if auto_save and output_dir:
        ensure_output_dir(output_dir)

    results = []
    total = len(audio_files)
    for i, audio_path in enumerate(audio_files):
        if batch_stop_flag:
            print("Batch processing interrupted by user.")
            results.append("⏹️ Batch processing interrupted by user.")
            break

        progress(i/total, desc=f"Processing {os.path.basename(audio_path)}")
        print(f"\n--- Processing file {i+1}/{total}: {audio_path} ---")
        try:
            response = process_single_file(
                audio_path, text_input, system_prompt, user_prompt,
                max_audio_sec, max_new_tokens, do_sample, temperature, repetition_penalty,
                console_progress, noise_reduction, use_stem_separation, target_stem,
                use_gpu_demucs, save_debug, recognition_model, use_segmentation,
                segment_duration, whisper_language
            )
        except Exception as e:
            response = f"⏹️ Processing error (possibly interrupted): {e}"

        if batch_stop_flag:
            print("Batch processing interrupted by user after current file.")
            results.append("⏹️ Batch processing interrupted by user after current file.")
            break

        if auto_save and not response.startswith("⚠") and not response.startswith("Error") and not response.startswith("⏹️"):
            model_type_for_save = current_model_type if recognition_model == "Qwen Omni" else "whisper"
            saved_path, save_msg = save_result(response, audio_path, output_dir, model_type_for_save)
            if saved_path:
                results.append(f"{os.path.basename(audio_path)} -> {saved_path}")
            else:
                results.append(f"{os.path.basename(audio_path)}: {save_msg}")
        else:
            results.append(f"{os.path.basename(audio_path)}:\n{response}\n")

    summary = "\n".join(results)
    return summary

# ----------------------------------------------------------------------
# Prompt reset functions
# ----------------------------------------------------------------------
def reset_system_prompt():
    return DEFAULT_SYSTEM_PROMPT

def reset_user_prompt():
    if current_model_type == "captioner":
        return "*Task* Describe this audio in detail. Provide only textual description."
    elif current_model_type == "transcriber":
        return "*Task* Transcribe this audio in detail. Provide structured lyrics with section tags."
    else:
        return ""

# ----------------------------------------------------------------------
# Stop generation
# ----------------------------------------------------------------------
def stop_generation():
    global current_stop_event, batch_stop_flag
    print(">>> stop_generation called, setting batch_stop_flag = True")
    batch_stop_flag = True
    if current_stop_event is not None:
        current_stop_event.set()
        return "⏹️ Stop signal sent (generation will stop at next step)"
    return "⏹️ No active generation, but batch processing will be stopped"

# ----------------------------------------------------------------------
# Prepare file for download
# ----------------------------------------------------------------------
def prepare_download(text, audio_path, output_dir, model_type):
    if not text or text.startswith("⚠") or text.startswith("Error") or text.startswith("⏹️"):
        return None, "No valid result to save"
    if audio_path and isinstance(audio_path, str) and os.path.exists(audio_path):
        base = os.path.splitext(os.path.basename(audio_path))[0]
    else:
        base = "result"
    suffix = "caption" if model_type == "captioner" else "transcript"
    filename = f"{base}_{suffix}.txt"
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return temp_path, f"✅ File ready for download: {filename}"
    except Exception as e:
        return None, f"❌ File creation error: {e}"

# ----------------------------------------------------------------------
# GRADIO INTERFACE
# ----------------------------------------------------------------------
initial_models = get_models(False)
model_choices = [(m["name"], m) for m in initial_models]

# Whisper languages
whisper_languages = ["auto", "ru", "en", "de", "fr", "es", "it", "pt", "pl", "uk", "zh", "ja", "ko"]
# Available Whisper versions
whisper_versions = ["openai/whisper-large-v3", "openai/whisper-large-v2"]

with gr.Blocks(title="🎵 ACE-step: Captioner & Transcriber") as demo:
    gr.Markdown("# 🎵 ACE-step: Captioner & Transcriber")
    gr.Markdown("Select a model, adjust settings, and upload audio.")

    with gr.Row():
        recognition_model = gr.Radio(choices=["Qwen Omni", "Whisper Large"], value="Qwen Omni", label="Recognition Model", scale=2)

        # Whisper elements (initially hidden)
        with gr.Column(scale=3, visible=False) as whisper_col:
            whisper_version = gr.Dropdown(choices=whisper_versions, value="openai/whisper-large-v3", label="Whisper version")
            whisper_language = gr.Dropdown(choices=whisper_languages, value="auto", label="Language (Whisper)")

        # Qwen elements (initially visible)
        with gr.Column(scale=3, visible=True) as qwen_col:
            model_dropdown = gr.Dropdown(choices=model_choices, label="Qwen Model")
            advanced_search = gr.Checkbox(label="🔍 Advanced Search", value=False)

        # Load buttons
        load_qwen_btn = gr.Button("📥 Load Qwen", variant="primary", scale=1, visible=True)
        unload_qwen_btn = gr.Button("⏹️ Unload Qwen", variant="secondary", scale=1, visible=True)
        load_whisper_btn = gr.Button("📥 Load Whisper", variant="primary", scale=1, visible=False)
        unload_whisper_btn = gr.Button("⏹️ Unload Whisper", variant="secondary", scale=1, visible=False)

    # Toggle UI based on selected model
    def toggle_ui(choice):
        show_qwen = choice == "Qwen Omni"
        show_whisper = choice == "Whisper Large"
        return (
            gr.update(visible=show_whisper),        # whisper_col
            gr.update(visible=show_qwen),            # qwen_col
            gr.update(visible=show_qwen),            # load_qwen_btn
            gr.update(visible=show_qwen),            # unload_qwen_btn
            gr.update(visible=show_whisper),         # load_whisper_btn
            gr.update(visible=show_whisper),         # unload_whisper_btn
        )

    recognition_model.change(
        fn=toggle_ui,
        inputs=recognition_model,
        outputs=[whisper_col, qwen_col, load_qwen_btn, unload_qwen_btn, load_whisper_btn, unload_whisper_btn]
    )

    def update_model_list(advanced):
        models = get_models(advanced)
        return gr.Dropdown(choices=[(m["name"], m) for m in models])

    advanced_search.change(
        fn=update_model_list,
        inputs=advanced_search,
        outputs=model_dropdown
    )

    status_text = gr.Textbox(label="Status", value="Model not loaded", interactive=False)

    with gr.Accordion("⚙️ Settings", open=False):
        # Qwen-specific settings (hidden when Whisper is selected)
        with gr.Column(visible=True) as qwen_settings:
            gr.Markdown("### Qwen Parameters")
            with gr.Row():
                dtype_choice = gr.Dropdown(choices=["float16", "bfloat16", "float32", "float8_e4m3fn", "float8_e5m2"],
                                           value="bfloat16", label="Precision")
                quantization_choice = gr.Radio(choices=["None", "8-bit", "4-bit"], value="None", label="Quantization")
            with gr.Row():
                use_flash = gr.Checkbox(value=True, label="Flash Attention 2")
                disable_talker = gr.Checkbox(value=True, label="Disable Talker")
            with gr.Row():
                device_map_strategy = gr.Dropdown(choices=["auto", "sequential", "balanced", "balanced_low_0", "custom"],
                                                 value="auto", label="device_map")
                custom_limits = gr.Textbox(label="Custom limits", placeholder="20GB,20GB,15GB")
                offload_folder = gr.Textbox(label="Offload folder (CPU)", placeholder="e.g. ./offload")
            with gr.Row():
                max_audio_sec = gr.Slider(0, 600, value=0, step=10, label="Max audio duration (s) (0 = no trim)")
                max_new_tokens = gr.Slider(128, 4096, 2048, step=64, label="Max new tokens")
            with gr.Row():
                do_sample = gr.Checkbox(value=False, label="do_sample")
                temperature = gr.Slider(0.0, 2.0, 0.1, step=0.1, label="Temperature")
                repetition_penalty = gr.Slider(1.0, 2.0, value=1.0, step=0.05, label="Repetition penalty")
            gr.Markdown("### Prompts (Qwen only)")
            with gr.Row():
                system_prompt_input = gr.Textbox(label="System prompt", value=DEFAULT_SYSTEM_PROMPT, lines=3, scale=4)
                reset_system_btn = gr.Button("↺", scale=1, min_width=50, elem_classes="square-btn")
            with gr.Row():
                user_prompt_input = gr.Textbox(label="Task prompt", value="", lines=2, scale=4)
                reset_user_btn = gr.Button("↺", scale=1, min_width=50, elem_classes="square-btn")

        # Common settings (always visible)
        gr.Markdown("### Common Settings")
        with gr.Row():
            auto_save = gr.Checkbox(value=False, label="Auto-save results")
        with gr.Row():
            output_dir = gr.Textbox(label="Output folder (empty = same as audio)", placeholder="", scale=4)
            browse_output_btn = gr.Button("📂", scale=1, min_width=50, elem_classes="square-btn")
            browse_output_btn.click(
                fn=browse_folder,
                inputs=output_dir,
                outputs=output_dir
            )            
            console_progress = gr.Checkbox(value=False, label="Show progress in console")
        with gr.Row():
            noise_reduction = gr.Checkbox(value=False, label="Noise reduction (noisereduce)")
        with gr.Row():
            use_segmentation = gr.Checkbox(value=False, label="Segmented processing (split into chunks)")
            segment_duration = gr.Slider(10, 60, value=30, step=5, label="Chunk length (s)")
        with gr.Row():
            use_stem_separation = gr.Checkbox(value=False, label="🎤 Stem separation (Demucs)")
            target_stem = gr.Dropdown(
                choices=["vocals", "instrumental", "drums", "bass", "other"],
                value="vocals",
                label="Target stem",
                interactive=False
            )
        with gr.Row():
            use_gpu_demucs = gr.Checkbox(value=False, label="🚀 Use GPU for Demucs (if available)")
        with gr.Row():
            debug_audio_checkbox = gr.Checkbox(value=False, label="💾 Save audio before model (debug)")

        # Toggle stem dropdown
        def toggle_stem_dropdown(use_stem):
            return gr.update(interactive=use_stem)
        use_stem_separation.change(
            fn=toggle_stem_dropdown,
            inputs=use_stem_separation,
            outputs=target_stem
        )

        # Hide Qwen settings when Whisper is selected
        def toggle_qwen_settings(choice):
            return gr.update(visible=(choice == "Qwen Omni"))
        recognition_model.change(
            fn=toggle_qwen_settings,
            inputs=recognition_model,
            outputs=qwen_settings
        )

    # Input mode switch
    gr.Markdown("### Input Mode")
    with gr.Row():
        input_mode = gr.Radio(choices=["Single file", "Multiple files", "Folder"], value="Single file", label="")
    with gr.Column(visible=True) as single_col:
        audio_input = gr.Audio(type="filepath", label="Audio file")
        text_input = gr.Textbox(label="Additional text", placeholder="e.g., focus on vocals", lines=2)
    with gr.Column(visible=False) as multi_col:
        file_input = gr.Files(label="Select audio files", file_types=[".wav", ".mp3", ".flac", ".m4a", ".ogg"])
    with gr.Column(visible=False) as folder_col:
        with gr.Row():
            folder_input = gr.Textbox(label="Path to audio folder", placeholder="C:/music/ or select via button", scale=4)
            browse_folder_btn = gr.Button("📂", scale=1, min_width=50, elem_classes="square-btn")
        browse_folder_btn.click(
            fn=browse_folder,
            inputs=folder_input,
            outputs=folder_input
        )

    def switch_mode(mode):
        return (
            gr.update(visible=(mode == "Single file")),
            gr.update(visible=(mode == "Multiple files")),
            gr.update(visible=(mode == "Folder"))
        )
    input_mode.change(fn=switch_mode, inputs=input_mode, outputs=[single_col, multi_col, folder_col])

    with gr.Row():
        task_btn = gr.Button(value="⬇️ Load a model first", variant="secondary", interactive=False, size="lg", scale=2)
        stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1)
        clear_cache_btn = gr.Button("🧹 Clear cache", variant="secondary", scale=1)

    output = gr.Textbox(label="Result", lines=15, interactive=False)

    with gr.Row():
        save_btn = gr.Button("💾 Prepare for download", variant="secondary")
        download_btn = gr.DownloadButton("💾 Download result", visible=False, variant="primary")
        clear_all_btn = gr.Button("🗑️ Clear all inputs", size="lg")

    # Qwen load
    def load_qwen_and_update(model_info, dtype, quant, flash, talker, device_map, limits, offload):
        return load_qwen_model(model_info, dtype, quant, flash, talker, device_map, limits, offload)

    load_qwen_btn.click(
        fn=load_qwen_and_update,
        inputs=[model_dropdown, dtype_choice, quantization_choice, use_flash, disable_talker,
                device_map_strategy, custom_limits, offload_folder],
        outputs=[task_btn, status_text, user_prompt_input]
    )

    unload_qwen_btn.click(
        fn=unload_qwen_model,
        inputs=[],
        outputs=status_text
    ).then(
        fn=lambda: gr.update(value="⬇️ Load a model first", variant="secondary", interactive=False),
        outputs=task_btn
    )

    # Whisper load
    def load_whisper_and_update(version):
        load_whisper_model(version)
        if whisper_pipeline is not None:
            return gr.update(value="🎤 Run Whisper", variant="primary", interactive=True), "✅ Whisper loaded"
        else:
            return gr.update(value="⬇️ Load error", variant="secondary", interactive=False), "❌ Failed to load Whisper"

    load_whisper_btn.click(
        fn=load_whisper_and_update,
        inputs=[whisper_version],
        outputs=[task_btn, status_text]
    )

    def unload_whisper_and_update():
        unload_whisper_model()
        return gr.update(value="⬇️ Load a model first", variant="secondary", interactive=False), "⏹️ Whisper unloaded"

    unload_whisper_btn.click(
        fn=unload_whisper_and_update,
        inputs=[],
        outputs=[task_btn, status_text]
    )

    reset_system_btn.click(fn=reset_system_prompt, outputs=system_prompt_input)
    reset_user_btn.click(fn=reset_user_prompt, outputs=user_prompt_input)

    clear_cache_btn.click(
        fn=lambda: ("🧹 Cache cleared", torch.cuda.empty_cache() or gc.collect()),
        inputs=[],
        outputs=status_text
    )

    # Main processing function
    def run_task(mode, audio_file, text, file_list, folder, system, user,
                 max_sec, max_tokens, sample, temp, rep_penalty,
                 auto_save, out_dir, console_prog, noise_red, use_seg, seg_dur,
                 use_stem, target_stem, use_gpu, save_debug, rec_model, whisper_lang):
        global batch_stop_flag
        batch_stop_flag = False
        print(">>> run_task: batch_stop_flag reset")

        # Model load check
        if rec_model == "Qwen Omni" and current_model is None:
            return "⚠️ Load a Qwen model first."
        if rec_model == "Whisper Large v3" and whisper_pipeline is None:
            return "⚠️ Load a Whisper model first."

        try:
            if mode == "Single file":
                if not audio_file:
                    return "⚠️ Select an audio file."
                response = process_single_file(
                    audio_file, text, system, user,
                    max_sec, max_tokens, sample, temp, rep_penalty,
                    console_prog, noise_red, use_stem, target_stem, use_gpu, save_debug,
                    rec_model, use_seg, seg_dur, whisper_lang
                )
                if auto_save and not response.startswith("⚠") and not response.startswith("Error") and not response.startswith("⏹️"):
                    model_type_for_save = current_model_type if rec_model == "Qwen Omni" else "whisper"
                    saved, msg = save_result(response, audio_file, out_dir, model_type_for_save)
                    if saved:
                        return response + f"\n\n{msg}"
                return response
            else:
                return batch_process(file_list if file_list else [],
                                     folder if mode == "Folder" else None,
                                     text, system, user,
                                     max_sec, max_tokens, sample, temp, rep_penalty,
                                     auto_save, out_dir, console_prog,
                                     noise_red, use_seg, seg_dur,
                                     use_stem, target_stem, use_gpu, save_debug,
                                     rec_model, whisper_lang)
        except Exception as e:
            return f"❌ Unexpected error: {e}"

    task_event = task_btn.click(
        fn=run_task,
        inputs=[input_mode, audio_input, text_input, file_input, folder_input,
                system_prompt_input, user_prompt_input,
                max_audio_sec, max_new_tokens, do_sample, temperature, repetition_penalty,
                auto_save, output_dir, console_progress,
                noise_reduction, use_segmentation, segment_duration,
                use_stem_separation, target_stem, use_gpu_demucs, debug_audio_checkbox,
                recognition_model, whisper_language],
        outputs=output
    )

    stop_btn.click(
        fn=stop_generation,
        inputs=[],
        outputs=status_text
    )

    # Manual download
    def handle_save(current_out, audio_path, out_dir, rec_model):
        if rec_model == "Qwen Omni" and current_model_type is None:
            return gr.update(), "Load a model first"
        model_type_for_save = current_model_type if rec_model == "Qwen Omni" else "whisper"
        file_path, msg = prepare_download(current_out, audio_path, out_dir, model_type_for_save)
        if file_path:
            return gr.update(value=file_path, visible=True), msg
        else:
            return gr.update(visible=False), msg

    save_btn.click(
        fn=handle_save,
        inputs=[output, audio_input, output_dir, recognition_model],
        outputs=[download_btn, status_text]
    )

    # Clear all inputs
    def clear_all():
        return (
            None, "", DEFAULT_SYSTEM_PROMPT, reset_user_prompt(),
            0, 2048, False, 0.1, 1.0, "",
            None, None, None, None,
            gr.update(visible=False),
            False, False, 30,
            False, "vocals",
            False,
            False,
            False,
            "auto",
            "openai/whisper-large-v3"
        )

    clear_all_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[audio_input, text_input, system_prompt_input, user_prompt_input,
                 max_audio_sec, max_new_tokens, do_sample, temperature, repetition_penalty, output_dir,
                 audio_input, file_input, folder_input, output,
                 download_btn,
                 noise_reduction, use_segmentation, segment_duration,
                 use_stem_separation, target_stem, use_gpu_demucs, debug_audio_checkbox,
                 recognition_model, whisper_language, whisper_version]
    )

    gr.Markdown("---\n*Console diagnostics.*")

if __name__ == "__main__":
    demo.launch(inbrowser=True, css="""
    .square-btn {
        width: 40px !important;
        height: 40px !important;
        min-width: 40px !important;
        max-width: 40px !important;
        padding: 0 !important;
        font-size: 1.2em;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex: none !important;
    }
    """)
