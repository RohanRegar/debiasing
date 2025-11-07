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
from data.cross_modal_dataset import load_clip_embeddings_from_json
from module.cross_modal import CrossModalAutoencoders, CrossModalClassifier


def register_feature_hook(model) -> Tuple[torch.utils.hooks.RemovableHandle, list]:
    """Register hook to capture pre-logit features."""
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
def evaluate_lff_inspired(
    data_dir: str,
    dataset_tag: str,
    model_tag: str,
    num_classes: int,
    debiased_model_path: str,
    cross_modal_ckpt_path: str,
    emb_json: str,
    batch_size: int = 64,
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
    lff_ckpt = torch.load(debiased_model_path, map_location=device_t)
    lff_state = lff_ckpt.get("state_dict", lff_ckpt)
    lff_model.load_state_dict(lff_state, strict=True)
    lff_model.eval()

    # Load cross-modal models from checkpoint
    cm_ckpt = torch.load(cross_modal_ckpt_path, map_location=device_t)
    h = cm_ckpt["hyperparameters"]
    
    # Infer fd_dim from model_tag
    if model_tag == "ResNet20":
        fd_dim = 64
    elif model_tag == "ResNet18":
        fd_dim = 512
    elif model_tag == "ResNet50":
        fd_dim = 2048
    else:
        raise ValueError(f"Unknown model_tag: {model_tag}")
    
    # Load CLIP embeddings to get ed_dim
    cifar10_names = [
        "airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"
    ]
    clip_embeddings = load_clip_embeddings_from_json(emb_json, cifar10_names)
    ed_dim = len(clip_embeddings[0])
    
    # Convert to tensors
    for k, v in clip_embeddings.items():
        clip_embeddings[k] = torch.tensor(v, dtype=torch.float32).to(device_t)
    
    # Initialize cross-modal autoencoder
    cross_model = CrossModalAutoencoders(
        fd_dim=fd_dim,
        ed_dim=ed_dim,
        latent_dim=int(h["latent_dim"]),
        fd_hidden_dims=list(h["fd_hidden_dims"]),
        ed_hidden_dims=list(h["ed_hidden_dims"]),
        dropout=float(h.get("dropout", 0.3))
    ).to(device_t)
    cross_model.load_state_dict(cm_ckpt["autoencoder_state_dict"])
    cross_model.eval()

    # Initialize classifier
    classifier = CrossModalClassifier(
        latent_dim=int(h["latent_dim"]),
        num_classes=num_classes,
        hidden_dims=list(h["clf_hidden_dims"]),
        dropout=float(h.get("dropout", 0.3))
    ).to(device_t)
    classifier.load_state_dict(cm_ckpt["classifier_state_dict"])
    classifier.eval()

    # Prepare hook on LfF final linear input
    handle, captured = register_feature_hook(lff_model)

    total = 0
    correct = 0
    
    try:
        for images, attrs in loader:
            images = images.to(device_t)
            labels = attrs[:, 0].to(device_t)
            
            # Get text embeddings for this batch
            ed_batch = []
            for label in labels:
                ed_batch.append(clip_embeddings[int(label.item())])
            ed_batch = torch.stack(ed_batch)
            
            # Extract features from LfF model
            captured.clear()
            _ = lff_model(images)
            fd_batch = captured[0].to(device_t)
            
            # Forward pass through cross-modal models
            ae_outputs = cross_model(fd_batch, ed_batch, return_latent=True)
            fl_batch = ae_outputs["fl"]
            logits = classifier(fl_batch)
            
            # Compute accuracy
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.size(0))
    
    finally:
        handle.remove()
    
    acc = correct / max(1, total)
    return acc


def main():
    p = argparse.ArgumentParser(description="Evaluate LfF-inspired cross-modal classifier.")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dataset_tag", required=True)
    p.add_argument("--model_tag", default="ResNet20")
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--debiased_model_path", required=True, help="Path to debiased LfF model.th")
    p.add_argument("--cross_modal_ckpt_path", required=True, help="Path to LfF-inspired cross-modal checkpoint")
    p.add_argument("--emb_json", required=True, help="Path to CLIP embeddings JSON")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    acc = evaluate_lff_inspired(
        data_dir=args.data_dir,
        dataset_tag=args.dataset_tag,
        model_tag=args.model_tag,
        num_classes=args.num_classes,
        debiased_model_path=args.debiased_model_path,
        cross_modal_ckpt_path=args.cross_modal_ckpt_path,
        emb_json=args.emb_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    print(f"LfF-inspired Cross-Modal Eval accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
