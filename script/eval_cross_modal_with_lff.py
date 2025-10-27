import argparse
import os
import sys
from typing import Tuple

import torch
from torch.utils.data import DataLoader

# Ensure project root in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.util import get_dataset
from module.util import get_model
from module.cross_modal import CrossModalAutoencoders, DebiasedClassifier


def register_feature_hook(model) -> Tuple[torch.utils.hooks.RemovableHandle, list]:
    captured = []

    def hook(module, inputs, output):
        captured.append(inputs[0].detach())

    handle = None
    if hasattr(model, "linear") and isinstance(model.linear, torch.nn.Module):
        handle = model.linear.register_forward_hook(hook)
    elif hasattr(model, "fc") and isinstance(model.fc, torch.nn.Module):
        handle = model.fc.register_forward_hook(hook)
    else:
        raise AttributeError("Model does not expose a 'linear' or 'fc' layer to hook for features.")
    
    return handle, captured


@torch.no_grad()
def evaluate(
    data_dir: str,
    dataset_tag: str,
    model_tag: str,
    num_classes: int,
    lff_model_path: str,
    cross_modal_ckpt_path: str,
    classifier_path: str,
    batch_size: int = 256,
    num_workers: int = 0,
    device: str = "cuda",
) -> float:
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    # Dataset (eval split)
    ds = get_dataset(
        dataset_tag,
        data_dir=data_dir,
        dataset_split="eval",
        transform_split="eval",
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Load debiased LfF model for feature extraction
    lff_model = get_model(model_tag, num_classes=num_classes).to(device_t)
    lff_ckpt = torch.load(lff_model_path, map_location=device_t)
    lff_state = lff_ckpt.get("state_dict", lff_ckpt)
    lff_model.load_state_dict(lff_state, strict=True)
    lff_model.eval()

    # Load cross-modal AE from checkpoint (has hyperparameters)
    cm_ckpt = torch.load(cross_modal_ckpt_path, map_location=device_t)
    h = cm_ckpt["hyperparameters"]
    cross_model = CrossModalAutoencoders(
        fd_dim=int(h["fd_dim"]),
        ed_dim=int(h["ed_dim"]),
        latent_dim=int(h["latent_dim"]),
        hidden_dims=list(h["hidden_dims"]),
    ).to(device_t)
    cross_model.load_state_dict(cm_ckpt["model_state_dict"])  # type: ignore[index]
    cross_model.eval()

    # Build classifier and load weights
    classifier = DebiasedClassifier(cross_model, num_classes=num_classes).to(device_t)
    clf_sd = torch.load(classifier_path, map_location=device_t)
    classifier.load_state_dict(clf_sd["state_dict"])
    classifier.eval()

    # Prepare hook on LfF final linear input
    handle, captured = register_feature_hook(lff_model)

    total = 0
    correct = 0
    for img, attr in loader:
        img = img.to(device_t)
        labels = attr[:, 0].to(device_t)
        captured.clear()
        _ = lff_model(img)
        fd_batch = captured[0].to(device_t)
        logits = classifier(fd_batch)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.size(0))

    handle.remove()
    acc = correct / max(1, total)
    return acc


def main():
    p = argparse.ArgumentParser(description="Evaluate cross-modal classifier using LfF debiased features.")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dataset_tag", required=True)
    p.add_argument("--model_tag", default="ResNet20")
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--lff_model_path", required=True, help="Path to LfF model.th (debiased)")
    p.add_argument("--cross_modal_ckpt_path", required=True, help="Path to AE checkpoint with hyperparameters")
    p.add_argument("--classifier_path", required=True, help="Path to trained classifier .pth")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    acc = evaluate(
        data_dir=args.data_dir,
        dataset_tag=args.dataset_tag,
        model_tag=args.model_tag,
        num_classes=args.num_classes,
        lff_model_path=args.lff_model_path,
        cross_modal_ckpt_path=args.cross_modal_ckpt_path,
        classifier_path=args.classifier_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    print(f"Eval accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()


