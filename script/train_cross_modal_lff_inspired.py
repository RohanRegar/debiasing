import argparse
import os
import sys
import logging
from datetime import datetime
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.util import get_dataset
from module.util import get_model
from data.cross_modal_dataset import load_clip_embeddings_from_json
from module.cross_modal import CrossModalAutoencoders, CrossModalClassifier


class DualLogger:
    """Logger that writes to both console and file simultaneously."""
    
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        
        # Create logger
        self.logger = logging.getLogger('lff_inspired_training')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate messages
        self.logger.propagate = False
    
    def info(self, message: str):
        """Log info message to both console and file."""
        self.logger.info(message)


class LfFInspiredCrossModalLoss(nn.Module):
    """
    LfF-inspired loss that weights samples based on biased vs debiased model performance.
    Similar to LfF's W(x) = CE(f_B(x), y) / (CE(f_B(x), y) + CE(f_D(x), y))
    """
    
    def __init__(
        self, 
        lambda_self: float = 1.0, 
        lambda_cross: float = 1.0,
        lambda_clf: float = 1.0,
        lambda_lff: float = 0.5  # Weight for LfF-inspired weighting
    ) -> None:
        super().__init__()
        self.lambda_self = float(lambda_self)
        self.lambda_cross = float(lambda_cross)
        self.lambda_clf = float(lambda_clf)
        self.lambda_lff = float(lambda_lff)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def compute_lff_weights(
        self, 
        debiased_logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute LfF-inspired weights based on debiased model confidence.
        Higher weights for samples where debiased model is less confident (harder samples).
        
        Args:
            debiased_logits: Logits from debiased LfF model [B, num_classes]
            labels: True labels [B]
        
        Returns:
            weights: Sample weights [B]
        """
        # Compute cross-entropy loss (no reduction)
        ce_debiased = self.ce_loss(debiased_logits, labels)  # [B]
        
        # Compute confidence (higher for easier samples)
        probs = torch.softmax(debiased_logits, dim=1)  # [B, num_classes]
        confidence = probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]
        
        # Weight inversely proportional to confidence (higher weight for harder samples)
        # Use entropy-based weighting: higher entropy = more uncertainty = higher weight
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # [B]
        max_entropy = torch.log(torch.tensor(probs.size(1), dtype=torch.float32, device=probs.device))
        normalized_entropy = entropy / max_entropy  # [B] in [0, 1]
        
        # Combine CE loss and entropy for weighting
        weights = (ce_debiased / ce_debiased.mean()) * (1 + normalized_entropy)  # [B]
        
        return weights
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        fd: torch.Tensor,
        ed: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        debiased_logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute LfF-inspired cross-modal loss with sample weighting.
        
        Args:
            outputs: Cross-modal autoencoder outputs
            fd: Visual features [B, fd_dim]
            ed: Text embeddings [B, ed_dim]
            logits: Classification logits [B, num_classes]
            labels: True labels [B]
            debiased_logits: Logits from debiased LfF model [B, num_classes]
        """
        batch_size = fd.size(0)
        
        # Compute LfF-inspired weights
        lff_weights = self.compute_lff_weights(debiased_logits, labels)  # [B]
        
        # Self-reconstruction losses
        fd_self_loss = self.mse_loss(outputs['fd_recon'], fd)  # [B, fd_dim]
        ed_self_loss = self.mse_loss(outputs['ed_recon'], ed)  # [B, ed_dim]
        
        # Cross-modal reconstruction losses
        fd_cross_loss = self.mse_loss(outputs['fd_recon_from_el'], fd)  # [B, fd_dim]
        ed_cross_loss = self.mse_loss(outputs['ed_recon_from_fl'], ed)  # [B, ed_dim]
        
        # Reduce to per-sample losses
        fd_self_loss = fd_self_loss.mean(dim=1)  # [B]
        ed_self_loss = ed_self_loss.mean(dim=1)  # [B]
        fd_cross_loss = fd_cross_loss.mean(dim=1)  # [B]
        ed_cross_loss = ed_cross_loss.mean(dim=1)  # [B]
        
        # Classification loss (per sample)
        clf_loss_per_sample = self.ce_loss(logits, labels)  # [B]
        
        # Apply LfF weights to all losses
        weighted_fd_self = lff_weights * fd_self_loss
        weighted_ed_self = lff_weights * ed_self_loss
        weighted_fd_cross = lff_weights * fd_cross_loss
        weighted_ed_cross = lff_weights * ed_cross_loss
        weighted_clf_loss = lff_weights * clf_loss_per_sample
        
        # Combine losses
        self_loss = weighted_fd_self.mean() + weighted_ed_self.mean()
        cross_loss = weighted_fd_cross.mean() + weighted_ed_cross.mean()
        recon_loss = self_loss + cross_loss
        clf_loss = weighted_clf_loss.mean()
        
        # Total loss
        total_loss = (
            self.lambda_self * self_loss + 
            self.lambda_cross * cross_loss + 
            self.lambda_clf * clf_loss
        )
        
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "self_loss": self_loss,
            "cross_loss": cross_loss,
            "clf_loss": clf_loss,
            "clf_loss_unweighted": clf_loss_per_sample.mean(),
            "fd_self_loss": weighted_fd_self.mean(),
            "ed_self_loss": weighted_ed_self.mean(),
            "ed_cross_loss": weighted_ed_cross.mean(),
            "fd_cross_loss": weighted_fd_cross.mean(),
            "lff_weights_mean": lff_weights.mean(),
            "lff_weights_std": lff_weights.std(),
        }


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


def get_optimized_hyperparameters(model_tag: str, fd_dim: int, ed_dim: int = 512) -> Dict[str, Any]:
    """Get optimized hyperparameters for LfF-inspired training."""
    
    if model_tag == "ResNet20":
        return {
            "latent_dim": 128,
            "fd_hidden_dims": [96],
            "ed_hidden_dims": [256],
            "clf_hidden_dims": [64, 32],
            "lambda_self": 1.2,
            "lambda_cross": 0.8,
            "lambda_clf": 2.0,
            "lambda_lff": 0.5,  # LfF weighting strength
            "epochs": 80,
            "lr": 1e-3,
            "dropout": 0.2,
            "batch_size": 64,  # Smaller for memory efficiency
            "grad_clip_norm": 2.0,
        }
    elif model_tag == "ResNet18":
        return {
            "latent_dim": 256,
            "fd_hidden_dims": [384, 320],
            "ed_hidden_dims": [384, 320],
            "clf_hidden_dims": [192, 128, 64],
            "lambda_self": 1.0,
            "lambda_cross": 1.0,
            "lambda_clf": 1.8,
            "lambda_lff": 0.4,
            "epochs": 60,
            "lr": 8e-4,
            "dropout": 0.3,
            "batch_size": 32,
            "grad_clip_norm": 2.0,
        }
    elif model_tag == "ResNet50":
        return {
            "latent_dim": 512,
            "fd_hidden_dims": [1536, 1024, 768],
            "ed_hidden_dims": [512, 512],
            "clf_hidden_dims": [384, 256, 128],
            "lambda_self": 0.8,
            "lambda_cross": 1.2,
            "lambda_clf": 1.5,
            "lambda_lff": 0.3,
            "epochs": 50,
            "lr": 5e-4,
            "dropout": 0.4,
            "batch_size": 16,
            "grad_clip_norm": 1.5,
        }
    else:
        raise ValueError(f"Unknown model_tag: {model_tag}")


def train_lff_inspired_cross_modal(
    autoencoder: CrossModalAutoencoders,
    classifier: CrossModalClassifier,
    dataloader: DataLoader,
    eval_dataloader: DataLoader,
    debiased_model: nn.Module,
    debiased_handle: torch.utils.hooks.RemovableHandle,
    debiased_captured: list,
    clip_embeddings: Dict[int, torch.Tensor],
    hyperparams: Dict[str, Any],
    logger: DualLogger,
    output_dir: str,
    device: str = "cuda"
) -> Tuple[CrossModalAutoencoders, CrossModalClassifier]:
    """Train cross-modal models with LfF-inspired sample weighting."""
    
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    autoencoder = autoencoder.to(device_t)
    classifier = classifier.to(device_t)
    
    # LfF-inspired loss function
    criterion = LfFInspiredCrossModalLoss(
        lambda_self=hyperparams["lambda_self"],
        lambda_cross=hyperparams["lambda_cross"],
        lambda_clf=hyperparams["lambda_clf"],
        lambda_lff=hyperparams["lambda_lff"]
    )
    
    # Joint optimizer
    all_params = list(autoencoder.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=hyperparams["lr"],
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=8
    )
    
    autoencoder.train()
    classifier.train()
    epochs = hyperparams["epochs"]
    
    logger.info(f"LfF-inspired training for {epochs} epochs...")
    logger.info(f"Loss weights - Self: {hyperparams['lambda_self']}, "
                f"Cross: {hyperparams['lambda_cross']}, "
                f"Classification: {hyperparams['lambda_clf']}, "
                f"LfF: {hyperparams['lambda_lff']}")
    
    best_eval_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        total_loss = 0.0
        total_recon_loss = 0.0
        total_clf_loss = 0.0
        total_lff_weights = 0.0
        train_correct = 0
        train_total = 0
        num_batches = 0
        
        for batch_idx, (images, attrs) in enumerate(dataloader):
            images = images.to(device_t)
            labels = attrs[:, 0].to(device_t)  # Target attribute (class labels)
            
            optimizer.zero_grad()
            
            # Get text embeddings for this batch
            ed_batch = []
            for label in labels:
                ed_batch.append(clip_embeddings[int(label.item())])
            ed_batch = torch.stack(ed_batch).to(device_t)
            
            # Forward pass through debiased LfF model to get features and logits
            debiased_captured.clear()
            
            debiased_logits = debiased_model(images)
            
            # Extract features (Fd) from debiased model
            fd_batch = debiased_captured[0].to(device_t)
            
            # Forward pass through cross-modal autoencoder
            ae_outputs = autoencoder(fd_batch, ed_batch, return_latent=True)
            fl_batch = ae_outputs["fl"]
            
            # Forward pass through classifier
            logits = classifier(fl_batch)
            
            # Compute LfF-inspired loss
            loss_dict = criterion(
                ae_outputs, fd_batch, ed_batch, logits, labels,
                debiased_logits
            )
            
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_clf_loss += loss_dict['clf_loss'].item()
            total_lff_weights += loss_dict['lff_weights_mean'].item()
            num_batches += 1
            
            # Compute accuracy
            pred = logits.argmax(dim=1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=hyperparams["grad_clip_norm"])
            
            optimizer.step()
            
            # Logging every 50 batches
            if batch_idx % 50 == 0:
                train_acc = train_correct / max(1, train_total)
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, "
                           f"Total: {loss_dict['total_loss']:.4f}, "
                           f"Recon: {loss_dict['recon_loss']:.4f}, "
                           f"Clf: {loss_dict['clf_loss']:.4f}, "
                           f"LfF_W: {loss_dict['lff_weights_mean']:.4f}±{loss_dict['lff_weights_std']:.4f}, "
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
            for images, attrs in eval_dataloader:
                images = images.to(device_t)
                labels = attrs[:, 0].to(device_t)
                
                # Get text embeddings
                ed_batch = []
                for label in labels:
                    ed_batch.append(clip_embeddings[int(label.item())])
                ed_batch = torch.stack(ed_batch).to(device_t)
                
                # Forward pass through debiased LfF model
                debiased_captured.clear()
                
                debiased_logits = debiased_model(images)
                fd_batch = debiased_captured[0].to(device_t)
                
                # Forward pass through cross-modal models
                ae_outputs = autoencoder(fd_batch, ed_batch, return_latent=True)
                fl_batch = ae_outputs["fl"]
                logits = classifier(fl_batch)
                
                # Compute loss
                loss_dict = criterion(
                    ae_outputs, fd_batch, ed_batch, logits, labels,
                    debiased_logits
                )
                eval_loss += loss_dict['total_loss'].item()
                eval_batches += 1
                
                # Compute accuracy
                pred = logits.argmax(dim=1)
                eval_correct += (pred == labels).sum().item()
                eval_total += labels.size(0)
        
        # Calculate metrics
        avg_train_loss = total_loss / max(1, num_batches)
        avg_recon_loss = total_recon_loss / max(1, num_batches)
        avg_clf_loss = total_clf_loss / max(1, num_batches)
        avg_lff_weights = total_lff_weights / max(1, num_batches)
        train_acc = train_correct / max(1, train_total)
        
        avg_eval_loss = eval_loss / max(1, eval_batches)
        eval_acc = eval_correct / max(1, eval_total)
        
        scheduler.step(avg_eval_loss)
        
        # Set back to training mode
        autoencoder.train()
        classifier.train()
        
        logger.info(f"Epoch {epoch+1}/{epochs} completed:")
        logger.info(f"  Train - Total: {avg_train_loss:.6f}, Recon: {avg_recon_loss:.6f}, "
                   f"Clf: {avg_clf_loss:.6f}, LfF_W: {avg_lff_weights:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"  Eval  - Total: {avg_eval_loss:.6f}, Acc: {eval_acc:.4f}")
        
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
            torch.save(checkpoint, f"{output_dir}/lff_inspired_best_model.pth")
            logger.info(f"  New best model saved! Eval Acc: {eval_acc:.4f}")
        
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
            os.makedirs(f"{output_dir}/lff_inspired_ckpts", exist_ok=True)
            torch.save(checkpoint, f"{output_dir}/lff_inspired_ckpts/lff_epoch_{epoch+1:03d}.pth")
    
    logger.info(f"LfF-inspired training completed! Best eval accuracy: {best_eval_acc:.4f}")
    return autoencoder, classifier


def main():
    parser = argparse.ArgumentParser(description="LfF-Inspired Cross-Modal Training")
    parser.add_argument("--data_dir", required=True, help="Root data directory")
    parser.add_argument("--dataset_tag", required=True, help="Dataset tag")
    parser.add_argument("--debiased_model_path", required=True, help="Path to debiased LfF model")
    parser.add_argument("--emb_json", required=True, help="Path to CLIP embeddings JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--model_tag", required=True, choices=["ResNet20", "ResNet18", "ResNet50"])
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f"lff_inspired_training_log_{timestamp}.txt")
    logger = DualLogger(log_file)
    
    logger.info("="*80)
    logger.info("LFF-INSPIRED CROSS-MODAL TRAINING SESSION STARTED")
    logger.info("="*80)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Model: {args.model_tag}")
    logger.info(f"Dataset: {args.dataset_tag}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info("")
    
    device_t = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = get_dataset(
        args.dataset_tag,
        data_dir=args.data_dir,
        dataset_split="train",
        transform_split="train"
    )
    eval_dataset = get_dataset(
        args.dataset_tag,
        data_dir=args.data_dir,
        dataset_split="eval",
        transform_split="eval"
    )
    
    logger.info(f"Training samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    # Load LfF debiased model
    logger.info("Loading LfF debiased model...")
    debiased_model = get_model(args.model_tag, num_classes=args.num_classes).to(device_t)
    
    # Load debiased model  
    debiased_ckpt = torch.load(args.debiased_model_path, map_location=device_t)
    debiased_state = debiased_ckpt.get("state_dict", debiased_ckpt)
    debiased_model.load_state_dict(debiased_state, strict=True)
    debiased_model.eval()
    
    # Register feature hook
    debiased_handle, debiased_captured = register_feature_hook(debiased_model)
    
    # Load CLIP embeddings
    logger.info("Loading CLIP embeddings...")
    cifar10_names = [
        "airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"
    ]
    clip_embeddings_tensor = load_clip_embeddings_from_json(args.emb_json, cifar10_names)
    
    # Convert to dictionary for easy access by class index
    clip_embeddings = {}
    for i in range(clip_embeddings_tensor.size(0)):
        clip_embeddings[i] = clip_embeddings_tensor[i]
    
    ed_dim = clip_embeddings_tensor.size(1)
    
    # Infer fd_dim from model
    if args.model_tag == "ResNet20":
        fd_dim = 64
    elif args.model_tag == "ResNet18":
        fd_dim = 512
    elif args.model_tag == "ResNet50":
        fd_dim = 2048
    
    logger.info(f"Visual feature dimension: {fd_dim}")
    logger.info(f"Text embedding dimension: {ed_dim}")
    
    # Get optimized hyperparameters
    hyperparams = get_optimized_hyperparameters(args.model_tag, fd_dim, ed_dim)
    logger.info(f"LfF-inspired training hyperparameters for {args.model_tag}:")
    for k, v in hyperparams.items():
        logger.info(f"  {k}: {v}")
    logger.info("")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=hyperparams["batch_size"], 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=hyperparams["batch_size"], 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    # Initialize models
    logger.info("Initializing cross-modal models...")
    autoencoder = CrossModalAutoencoders(
        fd_dim=fd_dim,
        ed_dim=ed_dim,
        latent_dim=hyperparams["latent_dim"],
        fd_hidden_dims=hyperparams["fd_hidden_dims"],
        ed_hidden_dims=hyperparams["ed_hidden_dims"],
        dropout=hyperparams["dropout"]
    )
    
    classifier = CrossModalClassifier(
        latent_dim=hyperparams["latent_dim"],
        num_classes=args.num_classes,
        hidden_dims=hyperparams["clf_hidden_dims"],
        dropout=hyperparams["dropout"]
    )
    
    logger.info(f"Model architecture:")
    logger.info(f"  Autoencoder:")
    logger.info(f"    fd_dim: {fd_dim} -> latent_dim: {hyperparams['latent_dim']}")
    logger.info(f"    fd_hidden_dims: {hyperparams['fd_hidden_dims']}")
    logger.info(f"    ed_dim: {ed_dim} -> latent_dim: {hyperparams['latent_dim']}")
    logger.info(f"    ed_hidden_dims: {hyperparams['ed_hidden_dims']}")
    logger.info(f"  Classifier:")
    logger.info(f"    latent_dim: {hyperparams['latent_dim']} -> num_classes: {args.num_classes}")
    logger.info(f"    clf_hidden_dims: {hyperparams['clf_hidden_dims']}")
    logger.info("")
    
    # LfF-inspired training
    logger.info("Starting LfF-inspired training...")
    try:
        autoencoder, classifier = train_lff_inspired_cross_modal(
            autoencoder, classifier, train_dataloader, eval_dataloader,
            debiased_model, debiased_handle, debiased_captured, clip_embeddings,
            hyperparams, logger, args.output_dir, args.device
        )
    finally:
        # Clean up hook
        debiased_handle.remove()
    
    # Save final models
    logger.info("")
    logger.info("Saving final models...")
    final_checkpoint = {
        'autoencoder_state_dict': autoencoder.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'hyperparameters': hyperparams
    }
    torch.save(final_checkpoint, f"{args.output_dir}/lff_inspired_final_model.pth")
    logger.info(f"Saved LfF-inspired final model: {args.output_dir}/lff_inspired_final_model.pth")
    
    logger.info("")
    logger.info("="*80)
    logger.info("LFF-INSPIRED CROSS-MODAL TRAINING SESSION COMPLETED")
    logger.info("="*80)


if __name__ == "__main__":
    main()
