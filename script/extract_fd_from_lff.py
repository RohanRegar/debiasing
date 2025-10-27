import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path when running as a script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.util import get_dataset
from module.util import get_model


def register_feature_hook(model) -> Tuple[torch.nn.Module, list]:
    """Register a hook to capture inputs to the final linear layer (pre-logit features)."""
    captured = []

    def hook(module, inputs, output):
        # inputs is a tuple; take the first (Tensor [B, F])
        captured.append(inputs[0].detach().cpu())

    handle = None
    if hasattr(model, "linear") and isinstance(model.linear, torch.nn.Module):
        handle = model.linear.register_forward_hook(hook)
    elif hasattr(model, "fc") and isinstance(model.fc, torch.nn.Module):
        handle = model.fc.register_forward_hook(hook)
    else:
        raise AttributeError("Model does not expose a 'linear' or 'fc' layer to hook for features.")

    return handle, captured


def extract_features(
    data_dir: str,
    dataset_tag: str,
    split: str,
    model_tag: str,
    num_classes: int,
    model_path: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    ds = get_dataset(dataset_tag, data_dir=data_dir, dataset_split=split, transform_split=split if split != "eval" else "eval")
    # ds returns (image, attr). target label index assumed 0
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = get_model(model_tag, num_classes=num_classes).to(device_t)
    ckpt = torch.load(model_path, map_location=device_t)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    handle, captured = register_feature_hook(model)

    labels = []
    with torch.no_grad():
        for img, attr in loader:
            img = img.to(device_t)
            label = attr[:, 0].to(device_t)
            _ = model(img)
            labels.append(label.detach().cpu())

    handle.remove()
    features = torch.cat(captured, dim=0).numpy()
    labels_np = torch.cat(labels, dim=0).numpy()
    return features, labels_np


def main():
    parser = argparse.ArgumentParser(description="Extract visual features (Fd) from trained LfF model.")
    parser.add_argument("--data_dir", required=True, help="Root data directory (e.g., /home/user/datasets/debias)")
    parser.add_argument("--dataset_tag", required=True, help="Dataset tag (e.g., CorruptedCIFAR10-Skewed0.01-Severity4-Type0)")
    parser.add_argument("--model_tag", default="ResNet20", help="Model tag used in training (e.g., ResNet20, MLP)")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--model_path", required=True, help="Path to saved model.th checkpoint")
    parser.add_argument("--output_dir", required=True, help="Directory to save fd/labels npy files")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for split in ["train", "eval"]:
        feats, labels = extract_features(
            data_dir=args.data_dir,
            dataset_tag=args.dataset_tag,
            split=split,
            model_tag=args.model_tag,
            num_classes=args.num_classes,
            model_path=args.model_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
        np.save(os.path.join(args.output_dir, f"fd_{split}.npy"), feats)
        np.save(os.path.join(args.output_dir, f"labels_{split}.npy"), labels)
        print(f"Saved {split} features to {args.output_dir}")


if __name__ == "__main__":
    main()


