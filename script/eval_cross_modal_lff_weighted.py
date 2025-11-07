import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.cross_modal_dataset import load_clip_embeddings_from_json
from module.cross_modal import CrossModalAutoencoders, CrossModalClassifier


@torch.no_grad()
def evaluate_lff_weighted(
    fd_eval_path: str,
    y_eval_path: str,
    cross_modal_ckpt_path: str,
    emb_json: str,
    batch_size: int = 128,
    device: str = "cuda",
) -> float:
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load pre-extracted features
    fd_eval_np = np.load(fd_eval_path).astype(np.float32)
    y_eval_np = np.load(y_eval_path).astype(np.int64)
    
    fd_eval_t = torch.from_numpy(fd_eval_np)
    y_eval_t = torch.from_numpy(y_eval_np)
    
    print(f"Loaded {len(fd_eval_np)} evaluation samples")
    print(f"Visual feature dimension: {fd_eval_np.shape[1]}")

    # Load cross-modal models from checkpoint
    cm_ckpt = torch.load(cross_modal_ckpt_path, map_location=device_t)
    h = cm_ckpt["hyperparameters"]
    
    fd_dim = fd_eval_np.shape[1]
    
    # Load CLIP embeddings
    cifar10_names = [
        "airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"
    ]
    clip_embeddings_tensor = load_clip_embeddings_from_json(emb_json, cifar10_names)
    ed_dim = clip_embeddings_tensor.size(1)
    
    print(f"Text embedding dimension: {ed_dim}")
    
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
        num_classes=10,  # CIFAR-10
        hidden_dims=list(h["clf_hidden_dims"]),
        dropout=float(h.get("dropout", 0.3))
    ).to(device_t)
    classifier.load_state_dict(cm_ckpt["classifier_state_dict"])
    classifier.eval()

    # Create evaluation dataloader
    eval_dataset = TensorDataset(fd_eval_t, y_eval_t)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    total = 0
    correct = 0
    
    for fd_batch, label_batch in eval_loader:
        fd_batch = fd_batch.to(device_t)
        label_batch = label_batch.to(device_t)
        
        # Get text embeddings for this batch
        ed_batch = []
        for label in label_batch:
            ed_batch.append(clip_embeddings_tensor[int(label.item())])
        ed_batch = torch.stack(ed_batch).to(device_t)
        
        # Forward pass through cross-modal models
        ae_outputs = cross_model(fd_batch, ed_batch, return_latent=True)
        fl_batch = ae_outputs["fl"]
        logits = classifier(fl_batch)
        
        # Compute accuracy
        pred = logits.argmax(dim=1)
        correct += int((pred == label_batch).sum().item())
        total += int(label_batch.size(0))
    
    acc = correct / max(1, total)
    return acc


def main():
    p = argparse.ArgumentParser(description="Evaluate LfF-weighted cross-modal classifier.")
    p.add_argument("--fd_eval", required=True, help="Path to eval visual features .npy")
    p.add_argument("--y_eval", required=True, help="Path to eval labels .npy")
    p.add_argument("--cross_modal_ckpt_path", required=True, help="Path to LfF-weighted cross-modal checkpoint")
    p.add_argument("--emb_json", required=True, help="Path to CLIP embeddings JSON")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    acc = evaluate_lff_weighted(
        fd_eval_path=args.fd_eval,
        y_eval_path=args.y_eval,
        cross_modal_ckpt_path=args.cross_modal_ckpt_path,
        emb_json=args.emb_json,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"LfF-weighted Cross-Modal Eval accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
