import argparse
import os
import sys
import logging
from datetime import datetime
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.util import get_dataset
from module.util import get_model
from data.cross_modal_dataset import build_dataloader_from_paths, load_clip_embeddings_from_json
from module.cross_modal import CrossModalAutoencoders, CrossModalClassifier


class DualLogger:
    """Logger that writes to both console and file simultaneously."""
    
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        
        # Create logger
        self.logger = logging.getLogger('lff_weighted_training')
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


class LfFWeightedCrossModalLoss(nn.Module):
    """
    Cross-modal loss with LfF-style sample weighting based on model confidence.
    Uses pre-computed sample weights from LfF model predictions.
    """
    
    def __init__(
        self, 
        lambda_self: float = 1.0, 
        lambda_cross: float = 1.0,
        lambda_clf: float = 1.0
    ) -> None:
        super().__init__()
        self.lambda_self = float(lambda_self)
        self.lambda_cross = float(lambda_cross)
        self.lambda_clf = float(lambda_clf)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        fd: torch.Tensor,
        ed: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted cross-modal loss.
        
        Args:
            outputs: Cross-modal autoencoder outputs
            fd: Visual features [B, fd_dim]
            ed: Text embeddings [B, ed_dim]
            logits: Classification logits [B, num_classes]
            labels: True labels [B]
            sample_weights: LfF-based sample weights [B]
        """
        batch_size = fd.size(0)
        
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
        weighted_fd_self = sample_weights * fd_self_loss
        weighted_ed_self = sample_weights * ed_self_loss
        weighted_fd_cross = sample_weights * fd_cross_loss
        weighted_ed_cross = sample_weights * ed_cross_loss
        weighted_clf_loss = sample_weights * clf_loss_per_sample
        
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
            "sample_weights_mean": sample_weights.mean(),
            "sample_weights_std": sample_weights.std(),
        }


def compute_lff_sample_weights(
    dataset_loader: DataLoader,
    lff_model: nn.Module,
    device: str = "cuda"
) -> Dict[int, float]:
    """
    Compute LfF-style sample weights based on model confidence/uncertainty.
    
    Args:
        dataset_loader: DataLoader for the original dataset (with images)
        lff_model: Trained LfF model
        device: Device to run computation on
    
    Returns:
        Dictionary mapping sample indices to weights
    """
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    lff_model.eval()
    
    sample_weights = {}
    
    with torch.no_grad():
        for batch_idx, (images, attrs) in enumerate(dataset_loader):
            images = images.to(device_t)
            labels = attrs[:, 0].to(device_t)  # Target attribute (class labels)
            
            # Get model predictions
            logits = lff_model(images)
            
            # Compute confidence and uncertainty
            probs = torch.softmax(logits, dim=1)  # [B, num_classes]
            
            # Method 1: Entropy-based uncertainty (higher entropy = more uncertain = higher weight)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # [B]
            max_entropy = torch.log(torch.tensor(probs.size(1), dtype=torch.float32, device=device_t))
            normalized_entropy = entropy / max_entropy  # [B] in [0, 1]
            
            # Method 2: Cross-entropy loss (higher loss = harder sample = higher weight)
            ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, labels)  # [B]
            
            # Method 3: Confidence (lower confidence = higher weight)
            confidence = probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]
            
            # Combine methods for final weights
            # Higher weight for: high entropy + high loss + low confidence
            weights = (
                (1 + normalized_entropy) *  # Entropy component
                (ce_loss / ce_loss.mean()) *  # Loss component  
                (2 - confidence)  # Confidence component (inverted)
            )
            
            # Normalize weights to have mean 1
            weights = weights / weights.mean()
            
            # Store weights for each sample in the batch
            batch_start_idx = batch_idx * dataset_loader.batch_size
            for i, weight in enumerate(weights):
                sample_idx = batch_start_idx + i
                sample_weights[sample_idx] = float(weight.cpu())
    
    return sample_weights


class WeightedDataset(torch.utils.data.Dataset):
    """Dataset that includes sample weights for LfF-weighted training."""
    
    def __init__(
        self, 
        fd_data: np.ndarray, 
        labels: np.ndarray, 
        clip_embeddings: Dict[int, torch.Tensor],
        sample_weights: Dict[int, float]
    ):
        self.fd_data = torch.from_numpy(fd_data).float()
        self.labels = torch.from_numpy(labels).long()
        self.clip_embeddings = clip_embeddings
        self.sample_weights = sample_weights
    
    def __len__(self):
        return len(self.fd_data)
    
    def __getitem__(self, idx):
        fd = self.fd_data[idx]
        label = self.labels[idx]
        ed = self.clip_embeddings[int(label.item())]
        weight = self.sample_weights.get(idx, 1.0)  # Default weight 1.0 if not found
        
        return fd, ed, label, torch.tensor(weight, dtype=torch.float32)


def get_optimized_hyperparameters(model_tag: str, fd_dim: int, ed_dim: int = 512) -> Dict[str, Any]:
    """Get optimized hyperparameters for LfF-weighted training."""
    
    if model_tag == "ResNet20":
        return {
            "latent_dim": 128,
            "fd_hidden_dims": [96],
            "ed_hidden_dims": [256],
            "clf_hidden_dims": [64, 32],
            "lambda_self": 1.0,
            "lambda_cross": 0.8,
            "lambda_clf": 2.0,
            "epochs": 100,
            "lr": 1e-3,
            "dropout": 0.2,
            "batch_size": 128,
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
            "epochs": 80,
            "lr": 8e-4,
            "dropout": 0.3,
            "batch_size": 64,
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
            "epochs": 60,
            "lr": 5e-4,
            "dropout": 0.4,
            "batch_size": 32,
            "grad_clip_norm": 1.5,
        }
    else:
        raise ValueError(f"Unknown model_tag: {model_tag}")


def train_lff_weighted_cross_modal(
    autoencoder: CrossModalAutoencoders,
    classifier: CrossModalClassifier,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    hyperparams: Dict[str, Any],
    logger: DualLogger,
    output_dir: str,
    device: str = "cuda"
) -> Tuple[CrossModalAutoencoders, CrossModalClassifier]:
    """Train cross-modal models with LfF-based sample weighting."""
    
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    autoencoder = autoencoder.to(device_t)
    classifier = classifier.to(device_t)
    
    # LfF-weighted loss function
    criterion = LfFWeightedCrossModalLoss(
        lambda_self=hyperparams["lambda_self"],
        lambda_cross=hyperparams["lambda_cross"],
        lambda_clf=hyperparams["lambda_clf"]
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
        optimizer, mode='min', factor=0.7, patience=10
    )
    
    autoencoder.train()
    classifier.train()
    epochs = hyperparams["epochs"]
    
    logger.info(f"LfF-weighted training for {epochs} epochs...")
    logger.info(f"Loss weights - Self: {hyperparams['lambda_self']}, "
                f"Cross: {hyperparams['lambda_cross']}, "
                f"Classification: {hyperparams['lambda_clf']}")
    
    best_eval_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        total_loss = 0.0
        total_recon_loss = 0.0
        total_clf_loss = 0.0
        total_sample_weights = 0.0
        train_correct = 0
        train_total = 0
        num_batches = 0
        
        for batch_idx, (fd_batch, ed_batch, label_batch, weight_batch) in enumerate(train_dataloader):
            fd_batch = fd_batch.to(device_t)
            ed_batch = ed_batch.to(device_t)
            label_batch = label_batch.to(device_t)
            weight_batch = weight_batch.to(device_t)
            
            optimizer.zero_grad()
            
            # Forward pass through cross-modal autoencoder
            ae_outputs = autoencoder(fd_batch, ed_batch, return_latent=True)
            fl_batch = ae_outputs["fl"]
            
            # Forward pass through classifier
            logits = classifier(fl_batch)
            
            # Compute weighted loss
            loss_dict = criterion(
                ae_outputs, fd_batch, ed_batch, logits, label_batch, weight_batch
            )
            
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_clf_loss += loss_dict['clf_loss'].item()
            total_sample_weights += loss_dict['sample_weights_mean'].item()
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
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, "
                           f"Total: {loss_dict['total_loss']:.4f}, "
                           f"Recon: {loss_dict['recon_loss']:.4f}, "
                           f"Clf: {loss_dict['clf_loss']:.4f}, "
                           f"Weights: {loss_dict['sample_weights_mean']:.4f}±{loss_dict['sample_weights_std']:.4f}, "
                           f"Train Acc: {train_acc:.4f}, "
                           f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Evaluation phase (without weights)
        autoencoder.eval()
        classifier.eval()
        eval_loss = 0.0
        eval_correct = 0
        eval_total = 0
        eval_batches = 0
        
        with torch.no_grad():
            for fd_batch, ed_batch, label_batch, _ in eval_dataloader:
                fd_batch = fd_batch.to(device_t)
                ed_batch = ed_batch.to(device_t)
                label_batch = label_batch.to(device_t)
                
                # Forward pass
                ae_outputs = autoencoder(fd_batch, ed_batch, return_latent=True)
                fl_batch = ae_outputs["fl"]
                logits = classifier(fl_batch)
                
                # Compute unweighted loss for evaluation
                eval_loss += nn.CrossEntropyLoss()(logits, label_batch).item()
                eval_batches += 1
                
                # Compute accuracy
                pred = logits.argmax(dim=1)
                eval_correct += (pred == label_batch).sum().item()
                eval_total += label_batch.size(0)
        
        # Calculate metrics
        avg_train_loss = total_loss / max(1, num_batches)
        avg_recon_loss = total_recon_loss / max(1, num_batches)
        avg_clf_loss = total_clf_loss / max(1, num_batches)
        avg_sample_weights = total_sample_weights / max(1, num_batches)
        train_acc = train_correct / max(1, train_total)
        
        avg_eval_loss = eval_loss / max(1, eval_batches)
        eval_acc = eval_correct / max(1, eval_total)
        
        scheduler.step(avg_eval_loss)
        
        # Set back to training mode
        autoencoder.train()
        classifier.train()
        
        logger.info(f"Epoch {epoch+1}/{epochs} completed:")
        logger.info(f"  Train - Total: {avg_train_loss:.6f}, Recon: {avg_recon_loss:.6f}, "
                   f"Clf: {avg_clf_loss:.6f}, Weights: {avg_sample_weights:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"  Eval  - Loss: {avg_eval_loss:.6f}, Acc: {eval_acc:.4f}")
        
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
            torch.save(checkpoint, f"{output_dir}/lff_weighted_best_model.pth")
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
            os.makedirs(f"{output_dir}/lff_weighted_ckpts", exist_ok=True)
            torch.save(checkpoint, f"{output_dir}/lff_weighted_ckpts/lff_weighted_epoch_{epoch+1:03d}.pth")
    
    logger.info(f"LfF-weighted training completed! Best eval accuracy: {best_eval_acc:.4f}")
    return autoencoder, classifier


def main():
    parser = argparse.ArgumentParser(description="LfF-Weighted Cross-Modal Training")
    parser.add_argument("--fd_train", required=True, help="Path to training visual features .npy")
    parser.add_argument("--y_train", required=True, help="Path to training labels .npy")
    parser.add_argument("--fd_eval", required=True, help="Path to eval visual features .npy")
    parser.add_argument("--y_eval", required=True, help="Path to eval labels .npy")
    parser.add_argument("--data_dir", required=True, help="Root data directory (for original images)")
    parser.add_argument("--dataset_tag", required=True, help="Dataset tag")
    parser.add_argument("--lff_model_path", required=True, help="Path to trained LfF model")
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
    log_file = os.path.join(args.output_dir, f"lff_weighted_training_log_{timestamp}.txt")
    logger = DualLogger(log_file)
    
    logger.info("="*80)
    logger.info("LFF-WEIGHTED CROSS-MODAL TRAINING SESSION STARTED")
    logger.info("="*80)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Model: {args.model_tag}")
    logger.info(f"Dataset: {args.dataset_tag}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info("")
    
    device_t = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load pre-extracted features
    logger.info("Loading pre-extracted features...")
    fd_train_np = np.load(args.fd_train).astype(np.float32)
    y_train_np = np.load(args.y_train).astype(np.int64)
    fd_eval_np = np.load(args.fd_eval).astype(np.float32)
    y_eval_np = np.load(args.y_eval).astype(np.int64)
    
    fd_dim = fd_train_np.shape[1]
    logger.info(f"Visual feature dimension: {fd_dim}")
    logger.info(f"Training samples: {len(fd_train_np)}, Eval samples: {len(fd_eval_np)}")
    
    # Load LfF model for weight computation
    logger.info("Loading LfF model for weight computation...")
    lff_model = get_model(args.model_tag, num_classes=args.num_classes).to(device_t)
    lff_ckpt = torch.load(args.lff_model_path, map_location=device_t)
    lff_state = lff_ckpt.get("state_dict", lff_ckpt)
    lff_model.load_state_dict(lff_state, strict=True)
    lff_model.eval()
    
    # Load original dataset for weight computation
    logger.info("Loading original dataset for weight computation...")
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
    
    # Compute sample weights
    logger.info("Computing LfF sample weights...")
    train_weight_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=0)
    eval_weight_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    train_sample_weights = compute_lff_sample_weights(train_weight_loader, lff_model, args.device)
    eval_sample_weights = compute_lff_sample_weights(eval_weight_loader, lff_model, args.device)
    
    logger.info(f"Computed weights for {len(train_sample_weights)} train samples, {len(eval_sample_weights)} eval samples")
    logger.info(f"Train weight stats - Mean: {np.mean(list(train_sample_weights.values())):.4f}, "
                f"Std: {np.std(list(train_sample_weights.values())):.4f}")
    
    # Load CLIP embeddings
    logger.info("Loading CLIP embeddings...")
    cifar10_names = [
        "airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"
    ]
    clip_embeddings_tensor = load_clip_embeddings_from_json(args.emb_json, cifar10_names)
    
    # Convert to dictionary
    clip_embeddings = {}
    for i in range(clip_embeddings_tensor.size(0)):
        clip_embeddings[i] = clip_embeddings_tensor[i]
    
    ed_dim = clip_embeddings_tensor.size(1)
    logger.info(f"Text embedding dimension: {ed_dim}")
    
    # Get optimized hyperparameters
    hyperparams = get_optimized_hyperparameters(args.model_tag, fd_dim, ed_dim)
    logger.info(f"LfF-weighted training hyperparameters for {args.model_tag}:")
    for k, v in hyperparams.items():
        logger.info(f"  {k}: {v}")
    logger.info("")
    
    # Create weighted datasets
    logger.info("Creating weighted datasets...")
    train_weighted_dataset = WeightedDataset(fd_train_np, y_train_np, clip_embeddings, train_sample_weights)
    eval_weighted_dataset = WeightedDataset(fd_eval_np, y_eval_np, clip_embeddings, eval_sample_weights)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_weighted_dataset, 
        batch_size=hyperparams["batch_size"], 
        shuffle=True, 
        num_workers=0
    )
    
    eval_dataloader = DataLoader(
        eval_weighted_dataset, 
        batch_size=hyperparams["batch_size"], 
        shuffle=False, 
        num_workers=0
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
    
    # LfF-weighted training
    logger.info("Starting LfF-weighted training...")
    autoencoder, classifier = train_lff_weighted_cross_modal(
        autoencoder, classifier, train_dataloader, eval_dataloader,
        hyperparams, logger, args.output_dir, args.device
    )
    
    # Save final models
    logger.info("")
    logger.info("Saving final models...")
    final_checkpoint = {
        'autoencoder_state_dict': autoencoder.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'hyperparameters': hyperparams
    }
    torch.save(final_checkpoint, f"{args.output_dir}/lff_weighted_final_model.pth")
    logger.info(f"Saved LfF-weighted final model: {args.output_dir}/lff_weighted_final_model.pth")
    
    logger.info("")
    logger.info("="*80)
    logger.info("LFF-WEIGHTED CROSS-MODAL TRAINING SESSION COMPLETED")
    logger.info("="*80)


if __name__ == "__main__":
    main()
