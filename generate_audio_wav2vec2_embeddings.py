import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import librosa
from transformers import AutoProcessor, Wav2Vec2Model


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def load_audio_mono_16k(path: str, target_sr: int = 16000) -> np.ndarray:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y


@torch.no_grad()
def embed_waveforms(
    waveforms: List[np.ndarray],
    processor,
    model,
    device: torch.device,
) -> np.ndarray:
    """
    Mean-pool the last hidden states over time to get a fixed-size embedding.
    Returns [num_files, hidden_size]
    """
    # Pack variable-length waveforms; processor does padding
    inputs = processor(
        waveforms,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)  # last_hidden_state: [B, T, H]
    feats = outputs.last_hidden_state  # [B, T, H]
    emb = feats.mean(dim=1)  # [B, H]
    return emb.cpu().numpy()


def collect_class_to_files(input_dir: Path) -> Dict[str, List[Path]]:
    """
    Supports:
      1) input_dir/<class>.wav (one file per class)
      2) input_dir/<class>/*.{wav,mp3,...} (many files per class)
    """
    class_to_files: Dict[str, List[Path]] = {}
    # Case 1: subdirs per class
    for entry in input_dir.iterdir():
        if entry.is_dir():
            files = [p for p in entry.glob("**/*") if p.suffix.lower() in AUDIO_EXTS]
            if files:
                class_to_files[entry.name] = files
    # Case 2: top-level files per class (filename stem is class name)
    top_files = [p for p in input_dir.glob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    for f in top_files:
        class_name = f.stem
        class_to_files.setdefault(class_name, []).append(f)
    return class_to_files


def main():
    ap = argparse.ArgumentParser(description="Generate Wav2Vec2 audio embeddings per class.")
    ap.add_argument("--input_dir", required=True, help="Folder with per-class audio: class.wav or class/*")
    ap.add_argument("--output_json", required=True, help="Path to write JSON {class_name: [embedding...]}")
    ap.add_argument("--model_name", default="facebook/wav2vec2-base-960h", help="HF model id")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Wav2Vec2Model.from_pretrained(args.model_name).to(device).eval()

    input_dir = Path(args.input_dir)
    class_to_files = collect_class_to_files(input_dir)
    if not class_to_files:
        raise RuntimeError(f"No audio files found under {input_dir}")

    result: Dict[str, List[float]] = {}
    for class_name, files in sorted(class_to_files.items()):
        waveforms = [load_audio_mono_16k(str(p)) for p in files]
        # For long clips, you can trim leading/trailing silence if needed with librosa.effects.trim
        emb_batch = embed_waveforms(waveforms, processor, model, device)  # [n, H]
        class_emb = emb_batch.mean(axis=0)  # average across multiple files for the class
        result[class_name] = class_emb.astype(np.float32).tolist()

    # Optional: print embedding dim
    first_key = next(iter(result))
    print("Embedding dim:", len(result[first_key]), "Num classes:", len(result))

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f)

    print(f"Wrote embeddings to {args.output_json}")


if __name__ == "__main__":
    main()