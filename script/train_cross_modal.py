import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.cross_modal_dataset import build_dataloader_from_paths, load_clip_embeddings_from_json
from module.cross_modal import (
    CrossModalAutoencoders,
    DebiasedClassifier,
    train_cross_modal_autoencoders,
)


def train_classifier(
    model: CrossModalAutoencoders,
    fd_train: torch.Tensor,
    y_train: torch.Tensor,
    fd_eval: Optional[torch.Tensor],
    y_eval: Optional[torch.Tensor],
    num_classes: int,
    device: str = "cuda",
    epochs: int = 20,
    lr: float = 1e-3,
) -> DebiasedClassifier:
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    clf = DebiasedClassifier(model, num_classes=num_classes).to(device_t)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-5)

    train_loader = DataLoader(
        TensorDataset(fd_train, y_train), batch_size=256, shuffle=True, num_workers=0
    )

    eval_loader = None
    if fd_eval is not None and y_eval is not None:
        eval_loader = DataLoader(
            TensorDataset(fd_eval, y_eval), batch_size=512, shuffle=False, num_workers=0
        )

    for epoch in range(1, epochs + 1):
        clf.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device_t)
            yb = yb.to(device_t)
            optimizer.zero_grad(set_to_none=True)
            logits = clf(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(xb.size(0))
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        msg = f"[Classifier][Epoch {epoch:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
        if eval_loader is not None:
            clf.eval()
            ecorr = 0
            etot = 0
            with torch.no_grad():
                for xb, yb in eval_loader:
                    xb = xb.to(device_t)
                    yb = yb.to(device_t)
                    logits = clf(xb)
                    pred = logits.argmax(dim=1)
                    ecorr += int((pred == yb).sum().item())
                    etot += int(xb.size(0))
            eval_acc = ecorr / max(1, etot)
            msg += f" eval_acc={eval_acc:.4f}"
        print(msg)

    return clf


def main():
    parser = argparse.ArgumentParser(description="Train cross-modal autoencoders and classifier.")
    parser.add_argument("--fd_train", required=True, help="Path to fd_train.npy")
    parser.add_argument("--y_train", required=True, help="Path to labels_train.npy")
    parser.add_argument("--fd_eval", required=False, help="Path to fd_eval.npy")
    parser.add_argument("--y_eval", required=False, help="Path to labels_eval.npy")
    parser.add_argument("--emb_json", required=True, help="Path to CLIP embeddings JSON")
    parser.add_argument("--output_dir", required=True, help="Directory to save checkpoints")
    parser.add_argument("--fd_dim", type=int, default=None, help="Visual feature dim; inferred if None")
    parser.add_argument("--ed_dim", type=int, default=None, help="Text embedding dim; inferred if None")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build training dataloader
    cifar10_names = [
        "airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"
    ]
    train_loader = build_dataloader_from_paths(
        args.fd_train, args.y_train, args.emb_json, cifar10_names, batch_size=256, shuffle=True
    )

    # Infer dims
    fd_sample, ed_sample, _ = next(iter(train_loader))
    fd_dim = int(fd_sample.size(1)) if args.fd_dim is None else args.fd_dim
    ed_dim = int(ed_sample.size(1)) if args.ed_dim is None else args.ed_dim
    
    model = CrossModalAutoencoders(
        fd_dim=fd_dim,
        ed_dim=ed_dim,
        latent_dim=args.latent_dim,
        hidden_dims=[512, 256],
    )

    # Train cross-modal AEs
    ckpt_dir = os.path.join(args.output_dir, "cross_modal_ckpts")
    train_cross_modal_autoencoders(
        model,
        train_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        ckpt_dir=ckpt_dir,
    )

    # Save final trained cross-modal model
    final_path = os.path.join(args.output_dir, "cross_modal_final.pth")
    torch.save({"state_dict": model.state_dict()}, final_path)
    print(f"Saved cross-modal model: {final_path}")

    # Optional classifier training if eval data provided
    fd_train_np = np.load(args.fd_train).astype(np.float32)
    y_train_np = np.load(args.y_train).astype(np.int64)
    fd_train_t = torch.from_numpy(fd_train_np)
    y_train_t = torch.from_numpy(y_train_np)

    fd_eval_t = y_eval_t = None
    if args.fd_eval and args.y_eval:
        fd_eval_np = np.load(args.fd_eval).astype(np.float32)
        y_eval_np = np.load(args.y_eval).astype(np.int64)
        fd_eval_t = torch.from_numpy(fd_eval_np)
        y_eval_t = torch.from_numpy(y_eval_np)

    clf = train_classifier(
        model,
        fd_train=fd_train_t,
        y_train=y_train_t,
        fd_eval=fd_eval_t,
        y_eval=y_eval_t,
        num_classes=args.num_classes,
        device=args.device,
        epochs=20,
        lr=1e-3,
    )

    clf_path = os.path.join(args.output_dir, "cross_modal_classifier.pth")
    torch.save({"state_dict": clf.state_dict()}, clf_path)
    print(f"Saved classifier: {clf_path}")


if __name__ == "__main__":
    main()


