import argparse
import json
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import torch
from transformers import AutoProcessor, ClapModel


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def load_audio_mono_48k(path: str, target_sr: int = 48000) -> np.ndarray:
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    return y


def collect_class_to_files(input_dir: Path) -> Dict[str, List[Path]]:
    """
    Supports either:
      1) input_dir/<class>.wav (or other audio) — one file per class
      2) input_dir/<class>/*.{wav,mp3,...} — multiple files per class
    """
    class_to_files: Dict[str, List[Path]] = {}

    # Case 1: subdirectories named by class
    for entry in input_dir.iterdir():
        if entry.is_dir():
            files = [p for p in entry.glob("**/*") if p.suffix.lower() in AUDIO_EXTS]
            if files:
                class_to_files[entry.name] = files

    # Case 2: top-level files named by class
    top_files = [p for p in input_dir.glob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    for f in top_files:
        class_name = f.stem
        class_to_files.setdefault(class_name, []).append(f)

    return class_to_files


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate CLAP (512-d) audio embeddings per class.")
    ap.add_argument("--input_dir", required=True, help="Folder with per-class audio: class.wav or class/*")
    ap.add_argument("--output_json", required=True, help="Path to write JSON {class_name: [embedding...]} ")
    ap.add_argument("--model_name", default="laion/clap-htsat-unfused", help="HF model id")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = ClapModel.from_pretrained(args.model_name).to(device).eval()

    input_dir = Path(args.input_dir)
    class_to_files = collect_class_to_files(input_dir)
    if not class_to_files:
        raise RuntimeError(f"No audio files found under {input_dir}")

    result: Dict[str, List[float]] = {}
    for class_name, files in sorted(class_to_files.items()):
        waveforms = [load_audio_mono_48k(str(p)) for p in files]
        inputs = processor(
            audios=waveforms,
            sampling_rate=48000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_audio_features(**inputs)  # [B, 512], L2-normalized
        emb = feats.mean(dim=0).cpu().numpy()      # average across files of a class
        result[class_name] = emb.tolist()

    first_key = next(iter(result))
    print("Embedding dim:", len(result[first_key]), "Num classes:", len(result))

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f)
    print(f"Wrote embeddings to {args.output_json}")


if __name__ == "__main__":
    main()


