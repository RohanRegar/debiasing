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
from module.cross_modal import CrossModalAutoencoders, JointCrossModalLoss, CrossModalClassifier


def get_optimized_hyperparameters(model_tag: str, fd_dim: int, ed_dim: int = 512) -> Dict[str, Any]:
    """Get optimized hyperparameters for each ResNet variant with joint training."""
    
    if model_tag == "ResNet20":
        # fd_dim = 64, ed_dim = 512
        return {
            "latent_dim": 128,  # Shared latent dimension
            "fd_hidden_dims": [128, 128],  # Visual path: 64 -> 128 -> 128 -> 128
            "ed_hidden_dims": [384, 256],  # Text path: 512 -> 384 -> 256 -> 128
            "clf_hidden_dims": [96, 64],  # Classifier: 128 -> 96 -> 64 -> 10
            "lambda_self": 1.5,  # Self-reconstruction weight
            "lambda_cross": 0.8,  # Cross-modal reconstruction weight
            "lambda_clf": 2.0,  # Classification loss weight (important for joint training)
            "epochs": 150,  # Joint training epochs
            "lr": 2e-3,  # Learning rate
            "dropout": 0.2,  # Less dropout for smaller model
            "batch_size": 128,  # Smaller batches
            "grad_clip_norm": 2.0,  # Gradient clipping
        }
    elif model_tag == "ResNet18":
        # fd_dim = 512, ed_dim = 512
        return {
            "latent_dim": 256,  # Shared latent dimension
            "fd_hidden_dims": [384, 320],  # Visual path: 512 -> 384 -> 320 -> 256
            "ed_hidden_dims": [384, 320],  # Text path: 512 -> 384 -> 320 -> 256
            "clf_hidden_dims": [192, 128, 64],  # Classifier: 256 -> 192 -> 128 -> 64 -> 10
            "lambda_self": 1.2,
            "lambda_cross": 1.0,
            "lambda_clf": 1.8,  # Classification loss weight
            "epochs": 120,
            "lr": 1e-3,
            "dropout": 0.3,
            "batch_size": 64,
            "grad_clip_norm": 2.0,
        }
    elif model_tag == "ResNet50":
        # fd_dim = 2048, ed_dim = 512
        return {
            "latent_dim": 512,  # Shared latent dimension
            "fd_hidden_dims": [1536, 1024, 768],  # Visual path: 2048 -> 1536 -> 1024 -> 768 -> 512
            "ed_hidden_dims": [512, 512],  # Text path: 512 -> 512 -> 512 -> 512
            "clf_hidden_dims": [384, 256, 128],  # Classifier: 512 -> 384 -> 256 -> 128 -> 10
            "lambda_self": 1.0,
            "lambda_cross": 1.2,
            "lambda_clf": 1.5,  # Classification loss weight
            "epochs": 100,
            "lr": 5e-4,  # Lower LR for larger model
            "dropout": 0.4,  # More dropout for larger model
            "batch_size": 32,  # Smaller batches for memory
            "grad_clip_norm": 1.5,
        }
    else:
        raise ValueError(f"Unknown model_tag: {model_tag}")


def train_joint_cross_modal(
    autoencoder: CrossModalAutoencoders,
    classifier: CrossModalClassifier,
    dataloader: DataLoader,
    eval_dataloader: DataLoader,
    hyperparams: Dict[str, Any],
    device: str = "cuda"
) -> Tuple[CrossModalAutoencoders, CrossModalClassifier]:
    """Train cross-modal autoencoders and classifier jointly."""
    
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    autoencoder = autoencoder.to(device_t)
    classifier = classifier.to(device_t)
    
    # Joint loss function with reconstruction and classification components
    criterion = JointCrossModalLoss(
        lambda_self=hyperparams["lambda_self"],
        lambda_cross=hyperparams["lambda_cross"],
        lambda_clf=hyperparams["lambda_clf"]
    )
    
    # Joint optimizer for both models
    all_params = list(autoencoder.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=hyperparams["lr"],
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=10, verbose=True
    )
    
    autoencoder.train()
    classifier.train()
    epochs = hyperparams["epochs"]
    
    print(f"Joint training for {epochs} epochs...")
    print(f"Loss weights - Self: {hyperparams['lambda_self']}, "
          f"Cross: {hyperparams['lambda_cross']}, "
          f"Classification: {hyperparams['lambda_clf']}")
    
    best_eval_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        total_loss = 0.0
        total_recon_loss = 0.0
        total_clf_loss = 0.0
        train_correct = 0
        train_total = 0
        num_batches = 0
        
        for batch_idx, (fd_batch, ed_batch, label_batch) in enumerate(dataloader):
            fd_batch = fd_batch.to(device_t)
            ed_batch = ed_batch.to(device_t)
            label_batch = label_batch.to(device_t)
            
            optimizer.zero_grad()
            
            # Forward pass through autoencoder (get latent features)
            ae_outputs = autoencoder(fd_batch, ed_batch, return_latent=True)
            fl_batch = ae_outputs["fl"]  # Visual latent features
            
            # Forward pass through classifier
            logits = classifier(fl_batch)
            
            # Compute joint loss
            loss_dict = criterion(ae_outputs, fd_batch, ed_batch, logits, label_batch)
            
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_clf_loss += loss_dict['clf_loss'].item()
            num_batches += 1
            
            # Compute accuracy
            pred = logits.argmax(dim=1)
            train_correct += (pred == label_batch).sum().item()
            train_total += label_batch.size(0)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=hyperparams["grad_clip_norm"])
            
            optimizer.step()
            
            # Logging every 50 batches
            if batch_idx % 50 == 0:
                train_acc = train_correct / max(1, train_total)
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, "
                      f"Total: {loss_dict['total_loss']:.4f}, "
                      f"Recon: {loss_dict['recon_loss']:.4f}, "
                      f"Clf: {loss_dict['clf_loss']:.4f}, "
                      f"Train Acc: {train_acc:.4f}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Evaluation phase
        autoencoder.eval()
        classifier.eval()
        eval_loss = 0.0
        eval_correct = 0
        eval_total = 0
        eval_batches = 0
        
        with torch.no_grad():
            for fd_batch, ed_batch, label_batch in eval_dataloader:
                fd_batch = fd_batch.to(device_t)
                ed_batch = ed_batch.to(device_t)
                label_batch = label_batch.to(device_t)
                
                # Forward pass
                ae_outputs = autoencoder(fd_batch, ed_batch, return_latent=True)
                fl_batch = ae_outputs["fl"]
                logits = classifier(fl_batch)
                
                # Compute loss
                loss_dict = criterion(ae_outputs, fd_batch, ed_batch, logits, label_batch)
                eval_loss += loss_dict['total_loss'].item()
                eval_batches += 1
                
                # Compute accuracy
                pred = logits.argmax(dim=1)
                eval_correct += (pred == label_batch).sum().item()
                eval_total += label_batch.size(0)
        
        # Calculate metrics
        avg_train_loss = total_loss / max(1, num_batches)
        avg_recon_loss = total_recon_loss / max(1, num_batches)
        avg_clf_loss = total_clf_loss / max(1, num_batches)
        train_acc = train_correct / max(1, train_total)
        
        avg_eval_loss = eval_loss / max(1, eval_batches)
        eval_acc = eval_correct / max(1, eval_total)
        
        scheduler.step(avg_eval_loss)
        
        # Set back to training mode
        autoencoder.train()
        classifier.train()
        
        print(f"Epoch {epoch+1}/{epochs} completed:")
        print(f"  Train - Total: {avg_train_loss:.6f}, Recon: {avg_recon_loss:.6f}, "
              f"Clf: {avg_clf_loss:.6f}, Acc: {train_acc:.4f}")
        print(f"  Eval  - Total: {avg_eval_loss:.6f}, Acc: {eval_acc:.4f}")
        
        # Save best model based on evaluation accuracy
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            checkpoint = {
                'epoch': epoch + 1,
                'autoencoder_state_dict': autoencoder.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'eval_accuracy': eval_acc,
                'hyperparameters': hyperparams
            }
            torch.save(checkpoint, f"{args.output_dir}/joint_best_model.pth")
            print(f"  New best model saved! Eval Acc: {eval_acc:.4f}")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'autoencoder_state_dict': autoencoder.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'eval_accuracy': eval_acc,
                'hyperparameters': hyperparams
            }
            os.makedirs(f"{args.output_dir}/joint_ckpts", exist_ok=True)
            torch.save(checkpoint, f"{args.output_dir}/joint_ckpts/joint_epoch_{epoch+1:03d}.pth")
    
    print(f"Joint training completed! Best eval accuracy: {best_eval_acc:.4f}")
    return autoencoder, classifier


def main():
    parser = argparse.ArgumentParser(description="Joint Cross-Modal and Classification Training")
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
    print(f"Training samples: {len(fd_train_np)}, Eval samples: {len(fd_eval_np)}")
    
    # Infer text embedding dimension
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
    print(f"Joint training hyperparameters for {args.model_tag}:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataloaders
    train_dataloader = build_dataloader_from_paths(
        args.fd_train, args.y_train, args.emb_json, cifar10_names,
        batch_size=hyperparams["batch_size"], shuffle=True
    )
    
    eval_dataloader = build_dataloader_from_paths(
        args.fd_eval, args.y_eval, args.emb_json, cifar10_names,
        batch_size=hyperparams["batch_size"], shuffle=False
    )
    
    print(f"Text embedding dimension: {ed_dim}")
    
    # Initialize autoencoder with optimized architecture
    autoencoder = CrossModalAutoencoders(
        fd_dim=fd_dim,
        ed_dim=ed_dim,
        latent_dim=hyperparams["latent_dim"],
        fd_hidden_dims=hyperparams["fd_hidden_dims"],
        ed_hidden_dims=hyperparams["ed_hidden_dims"],
        dropout=hyperparams["dropout"]
    )
    
    # Initialize classifier
    classifier = CrossModalClassifier(
        latent_dim=hyperparams["latent_dim"],
        num_classes=args.num_classes,
        hidden_dims=hyperparams["clf_hidden_dims"],
        dropout=hyperparams["dropout"]
    )
    
    print(f"Model architecture:")
    print(f"  Autoencoder:")
    print(f"    fd_dim: {fd_dim} -> latent_dim: {hyperparams['latent_dim']}")
    print(f"    fd_hidden_dims: {hyperparams['fd_hidden_dims']}")
    print(f"    ed_dim: {ed_dim} -> latent_dim: {hyperparams['latent_dim']}")
    print(f"    ed_hidden_dims: {hyperparams['ed_hidden_dims']}")
    print(f"  Classifier:")
    print(f"    latent_dim: {hyperparams['latent_dim']} -> num_classes: {args.num_classes}")
    print(f"    clf_hidden_dims: {hyperparams['clf_hidden_dims']}")
    
    # Joint training
    autoencoder, classifier = train_joint_cross_modal(
        autoencoder, classifier, train_dataloader, eval_dataloader, 
        hyperparams, args.device
    )
    
    # Save final models
    final_checkpoint = {
        'autoencoder_state_dict': autoencoder.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'hyperparameters': hyperparams
    }
    torch.save(final_checkpoint, f"{args.output_dir}/joint_final_model.pth")
    print(f"Saved joint final model: {args.output_dir}/joint_final_model.pth")


if __name__ == "__main__":
    main()

