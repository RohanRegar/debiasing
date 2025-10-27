import argparse
import os
import sys
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.cross_modal_dataset import build_dataloader_from_paths
from module.cross_modal import CrossModalAutoencoders, CrossModalLoss, DebiasedClassifier


def get_optimized_hyperparameters(model_tag: str, fd_dim: int, ed_dim: int = 512) -> Dict[str, Any]:
    """Get optimized hyperparameters for each ResNet variant with separate hidden dims."""
    
    if model_tag == "ResNet20":
        # fd_dim = 64, ed_dim = 512
        return {
            "latent_dim": 128,  # Shared latent dimension
            "fd_hidden_dims": [128, 128],  # Visual path: 64 -> 128 -> 128 -> 128
            "ed_hidden_dims": [384, 256],  # Text path: 512 -> 384 -> 256 -> 128
            "lambda_self": 1.5,  # Emphasize self-reconstruction
            "lambda_cross": 0.8,  # Reduce cross-modal weight
            "ae_epochs": 150,  # More training for smaller model
            "ae_lr": 2e-3,  # Higher learning rate
            "clf_epochs": 50,  # More classifier training
            "clf_lr": 1e-3,
            "dropout": 0.2,  # Less dropout for smaller model
            "batch_size": 128,  # Smaller batches
        }
    elif model_tag == "ResNet18":
        # fd_dim = 512, ed_dim = 512
        return {
            "latent_dim": 256,  # Shared latent dimension
            "fd_hidden_dims": [384, 320],  # Visual path: 512 -> 384 -> 320 -> 256
            "ed_hidden_dims": [384, 320],  # Text path: 512 -> 384 -> 320 -> 256
            "lambda_self": 1.2,
            "lambda_cross": 1.0,
            "ae_epochs": 120,
            "ae_lr": 1e-3,
            "clf_epochs": 40,
            "clf_lr": 8e-4,
            "dropout": 0.3,
            "batch_size": 64,
        }
    elif model_tag == "ResNet50":
        # fd_dim = 2048, ed_dim = 512
        return {
            "latent_dim": 512,  # Shared latent dimension
            "fd_hidden_dims": [1536, 1024, 768],  # Visual path: 2048 -> 1536 -> 1024 -> 768 -> 512
            "ed_hidden_dims": [512, 512],  # Text path: 512 -> 512 -> 512 -> 512 (maintain richness)
            "lambda_self": 1.0,
            "lambda_cross": 1.2,  # Emphasize cross-modal for rich features
            "ae_epochs": 100,
            "ae_lr": 5e-4,  # Lower LR for larger model
            "clf_epochs": 30,
            "clf_lr": 5e-4,
            "dropout": 0.4,  # More dropout for larger model
            "batch_size": 32,  # Smaller batches for memory
        }
    else:
        raise ValueError(f"Unknown model_tag: {model_tag}")


def train_cross_modal_optimized(
    model: CrossModalAutoencoders,
    dataloader: DataLoader,
    hyperparams: Dict[str, Any],
    device: str = "cuda"
) -> CrossModalAutoencoders:
    """Train cross-modal autoencoders with optimized hyperparameters."""
    
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device_t)
    
    # Optimized loss function with architecture-specific weights
    criterion = CrossModalLoss(
        lambda_cross=hyperparams["lambda_cross"],
        lambda_self=hyperparams["lambda_self"]
    )
    
    # Optimized optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams["ae_lr"],
        weight_decay=1e-4,  # Add weight decay for regularization
        betas=(0.9, 0.999)
    )
    
    # More aggressive scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=8
    )
    
    model.train()
    epochs = hyperparams["ae_epochs"]
    
    print(f"Training autoencoder for {epochs} epochs with optimized hyperparameters...")
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (fd_batch, ed_batch, _) in enumerate(dataloader):
            fd_batch = fd_batch.to(device_t)
            ed_batch = ed_batch.to(device_t)
            
            optimizer.zero_grad()
            outputs = model(fd_batch, ed_batch)
            loss_dict = criterion(outputs, fd_batch, ed_batch)
            
            total_loss += loss_dict['total_loss'].item()
            num_batches += 1
            
            loss_dict['total_loss'].backward()
            
            # Gradient clipping with adaptive norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            
            # Enhanced logging every 50 batches
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, "
                      f"Total: {loss_dict['total_loss']:.4f}, "
                      f"Self: {loss_dict['self_loss']:.4f}, "
                      f"Cross: {loss_dict['cross_loss']:.4f}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        avg_loss = total_loss / max(1, num_batches)
        scheduler.step(avg_loss)
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'hyperparameters': hyperparams
            }
            os.makedirs(f"{args.output_dir}/optimized_ckpts", exist_ok=True)
            torch.save(checkpoint, f"{args.output_dir}/optimized_ckpts/cross_modal_epoch_{epoch+1:03d}.pth")
        
        print(f"Epoch {epoch+1}/{epochs} completed. Avg Loss: {avg_loss:.6f}")
    
    return model


def train_classifier_optimized(
    cross_modal_model: CrossModalAutoencoders,
    fd_train: torch.Tensor,
    y_train: torch.Tensor,
    fd_eval: torch.Tensor,
    y_eval: torch.Tensor,
    hyperparams: Dict[str, Any],
    num_classes: int,
    device: str = "cuda"
) -> DebiasedClassifier:
    """Train classifier with optimized hyperparameters and validation."""
    
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Create classifier with optimized dropout
    classifier = DebiasedClassifier(
        cross_modal_model, 
        num_classes=num_classes, 
        dropout=hyperparams["dropout"]
    ).to(device_t)
    
    # Create optimized dataloaders
    train_dataset = TensorDataset(fd_train, y_train)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=hyperparams["batch_size"], 
        shuffle=True, 
        pin_memory=True
    )
    
    eval_dataset = TensorDataset(fd_eval, y_eval)
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=hyperparams["batch_size"], 
        shuffle=False, 
        pin_memory=True
    )
    
    # Optimized loss and optimizer for classification
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for regularization
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=hyperparams["clf_lr"],
        weight_decay=1e-3,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hyperparams["clf_epochs"], eta_min=1e-6
    )
    
    epochs = hyperparams["clf_epochs"]
    best_acc = 0.0
    
    print(f"Training classifier for {epochs} epochs with validation...")
    
    for epoch in range(epochs):
        # Training phase
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for fd_batch, y_batch in train_loader:
            fd_batch = fd_batch.to(device_t)
            y_batch = y_batch.to(device_t)
            
            optimizer.zero_grad()
            logits = classifier(fd_batch)
            loss = criterion(logits, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == y_batch).sum().item()
            train_total += y_batch.size(0)
        
        # Validation phase
        classifier.eval()
        eval_loss = 0.0
        eval_correct = 0
        eval_total = 0
        
        with torch.no_grad():
            for fd_batch, y_batch in eval_loader:
                fd_batch = fd_batch.to(device_t)
                y_batch = y_batch.to(device_t)
                
                logits = classifier(fd_batch)
                loss = criterion(logits, y_batch)
                
                eval_loss += loss.item()
                pred = logits.argmax(dim=1)
                eval_correct += (pred == y_batch).sum().item()
                eval_total += y_batch.size(0)
        
        train_acc = train_correct / train_total
        eval_acc = eval_correct / eval_total
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Eval Loss: {eval_loss/len(eval_loader):.4f}, "
              f"Eval Acc: {eval_acc:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save({
                'state_dict': classifier.state_dict(),
                'accuracy': eval_acc,
                'epoch': epoch + 1
            }, f"{args.output_dir}/optimized_classifier_best.pth")
    
    print(f"Best validation accuracy: {best_acc:.4f}")
    return classifier


def main():
    parser = argparse.ArgumentParser(description="Optimized Cross-Modal Training")
    parser.add_argument("--fd_train", required=True, help="Path to training visual features")
    parser.add_argument("--y_train", required=True, help="Path to training labels")
    parser.add_argument("--fd_eval", required=True, help="Path to eval visual features")
    parser.add_argument("--y_eval", required=True, help="Path to eval labels")
    parser.add_argument("--emb_json", required=True, help="Path to CLIP embeddings JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--model_tag", required=True, choices=["ResNet20", "ResNet18", "ResNet50"])
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    
    global args
    args = parser.parse_args()
    
    # Load data
    fd_train_np = np.load(args.fd_train).astype(np.float32)
    y_train_np = np.load(args.y_train).astype(np.int64)
    fd_eval_np = np.load(args.fd_eval).astype(np.float32)
    y_eval_np = np.load(args.y_eval).astype(np.int64)
    
    fd_dim = fd_train_np.shape[1]
    print(f"Visual feature dimension: {fd_dim}")
    
    # Infer text embedding dimension from dataloader first
    cifar10_names = [
        "airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"
    ]
    temp_dataloader = build_dataloader_from_paths(
        args.fd_train, args.y_train, args.emb_json, cifar10_names,
        batch_size=32, shuffle=False
    )
    fd_sample, ed_sample, _ = next(iter(temp_dataloader))
    ed_dim = int(ed_sample.size(1))
    
    # Get optimized hyperparameters for this architecture
    hyperparams = get_optimized_hyperparameters(args.model_tag, fd_dim, ed_dim)
    print(f"Optimized hyperparameters for {args.model_tag}:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create optimized dataloader using the existing function
    dataloader = build_dataloader_from_paths(
        args.fd_train, args.y_train, args.emb_json, cifar10_names,
        batch_size=hyperparams["batch_size"], shuffle=True
    )
    
    print(f"Text embedding dimension: {ed_dim}")
    
    # Initialize model with optimized architecture using separate hidden dimensions
    model = CrossModalAutoencoders(
        fd_dim=fd_dim,
        ed_dim=ed_dim,
        latent_dim=hyperparams["latent_dim"],
        fd_hidden_dims=hyperparams["fd_hidden_dims"],
        ed_hidden_dims=hyperparams["ed_hidden_dims"]
    )
    
    print(f"Model architecture:")
    print(f"  fd_dim: {fd_dim} -> latent_dim: {hyperparams['latent_dim']}")
    print(f"  fd_hidden_dims: {hyperparams['fd_hidden_dims']}")
    print(f"  ed_dim: {ed_dim} -> latent_dim: {hyperparams['latent_dim']}")
    print(f"  ed_hidden_dims: {hyperparams['ed_hidden_dims']}")
    
    # Train autoencoder with optimized parameters
    model = train_cross_modal_optimized(model, dataloader, hyperparams, args.device)
    
    # Save final autoencoder
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'hyperparameters': hyperparams
    }
    torch.save(final_checkpoint, f"{args.output_dir}/optimized_autoencoder_final.pth")
    print(f"Saved optimized autoencoder: {args.output_dir}/optimized_autoencoder_final.pth")
    
    # Convert to tensors for classifier training
    fd_train_t = torch.from_numpy(fd_train_np)
    y_train_t = torch.from_numpy(y_train_np)
    fd_eval_t = torch.from_numpy(fd_eval_np)
    y_eval_t = torch.from_numpy(y_eval_np)
    
    # Train classifier with optimized parameters
    classifier = train_classifier_optimized(
        model, fd_train_t, y_train_t, fd_eval_t, y_eval_t,
        hyperparams, args.num_classes, args.device
    )
    
    # Save final classifier
    torch.save({
        'state_dict': classifier.state_dict(),
        'hyperparameters': hyperparams
    }, f"{args.output_dir}/optimized_classifier_final.pth")
    print(f"Saved optimized classifier: {args.output_dir}/optimized_classifier_final.pth")


if __name__ == "__main__":
    main()
