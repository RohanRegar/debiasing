import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CrossModalAutoencoders(nn.Module):
    """
    Cross-modal autoencoders correlating visual features (Fd) and text embeddings (Ed).

    Encoders map to a shared latent dimensionality; decoders reconstruct each modality.
    Shared decoders are used for both self- and cross-modal reconstruction to enforce
    latent space compatibility.

    Args:
        fd_dim: Dimension of visual features Fd (e.g., 2048 for ResNet features)
        ed_dim: Dimension of text embeddings Ed (e.g., 512 for CLIP)
        latent_dim: Dimension of latent spaces Fl and El (shared)
        fd_hidden_dims: Hidden layer sizes for visual encoders/decoders (e.g., [512, 256])
        ed_hidden_dims: Hidden layer sizes for text encoders/decoders (e.g., [384, 256])
        dropout: Dropout probability applied after hidden layers
    """

    def __init__(
        self,
        fd_dim: int,
        ed_dim: int,
        latent_dim: int,
        fd_hidden_dims: List[int],
        ed_hidden_dims: List[int],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fd_dim = fd_dim
        self.ed_dim = ed_dim
        self.latent_dim = latent_dim
        self.fd_hidden_dims = list(fd_hidden_dims)
        self.ed_hidden_dims = list(ed_hidden_dims)
        self.dropout = dropout

        # Visual path with separate hidden dimensions
        self.encoder1 = self._build_encoder(fd_dim, latent_dim, self.fd_hidden_dims, dropout)
        self.decoder1 = self._build_decoder(latent_dim, fd_dim, self.fd_hidden_dims, dropout)

        # Text path with separate hidden dimensions
        self.encoder2 = self._build_encoder(ed_dim, latent_dim, self.ed_hidden_dims, dropout)
        self.decoder2 = self._build_decoder(latent_dim, ed_dim, self.ed_hidden_dims, dropout)

    def _build_encoder(
        self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        return nn.Sequential(*layers)

    def _build_decoder(
        self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        prev = input_dim
        # Reverse hidden dimensions for decoder (latent -> hidden -> output)
        for h in reversed(hidden_dims):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        return nn.Sequential(*layers)

    def forward(self, fd: torch.Tensor, ed: torch.Tensor, return_latent: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing self- and cross-modal reconstructions.

        Args:
            fd: Visual features tensor [batch_size, fd_dim]
            ed: Text embeddings tensor [batch_size, ed_dim]
            return_latent: If True, include latent features in output (default: True)

        Returns:
            Dict with the following keys mapping to tensors:
                - 'fl': visual latent [B, latent_dim] (if return_latent=True)
                - 'el': text latent [B, latent_dim] (if return_latent=True)
                - 'fd_recon': self-reconstructed visual [B, fd_dim]
                - 'ed_recon': self-reconstructed text [B, ed_dim]
                - 'ed_recon_from_fl': cross reconstruction visual->text [B, ed_dim]
                - 'fd_recon_from_el': cross reconstruction text->visual [B, fd_dim]
        """
        # Self reconstruction
        fl = self.encoder1(fd)
        fd_recon = self.decoder1(fl)

        el = self.encoder2(ed)
        ed_recon = self.decoder2(el)

        # Cross-modal reconstruction using shared decoders
        ed_recon_from_fl = self.decoder2(fl)
        fd_recon_from_el = self.decoder1(el)

        result = {
            "fd_recon": fd_recon,
            "ed_recon": ed_recon,
            "ed_recon_from_fl": ed_recon_from_fl,
            "fd_recon_from_el": fd_recon_from_el,
        }
        
        if return_latent:
            result["fl"] = fl
            result["el"] = el
            
        return result


class CrossModalLoss(nn.Module):
    """
    Composite loss for self- and cross-modal reconstruction.

    Args:
        lambda_cross: Weight for cross-modal reconstruction loss terms
        lambda_self: Weight for self reconstruction loss terms
    """

    def __init__(self, lambda_cross: float = 1.0, lambda_self: float = 1.0) -> None:
        super().__init__()
        self.lambda_cross = float(lambda_cross)
        self.lambda_self = float(lambda_self)
        self.mse_loss = nn.MSELoss()

    def forward(
        self, outputs: Dict[str, torch.Tensor], fd: torch.Tensor, ed: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        fd_self_loss = self.mse_loss(outputs["fd_recon"], fd)
        ed_self_loss = self.mse_loss(outputs["ed_recon"], ed)

        ed_cross_loss = self.mse_loss(outputs["ed_recon_from_fl"], ed)
        fd_cross_loss = self.mse_loss(outputs["fd_recon_from_el"], fd)

        self_loss = fd_self_loss + ed_self_loss
        cross_loss = ed_cross_loss + fd_cross_loss
        total_loss = self.lambda_self * self_loss + self.lambda_cross * cross_loss

        return {
            "total_loss": total_loss,
            "self_loss": self_loss,
            "cross_loss": cross_loss,
            "fd_self_loss": fd_self_loss,
            "ed_self_loss": ed_self_loss,
            "ed_cross_loss": ed_cross_loss,
            "fd_cross_loss": fd_cross_loss,
        }


class JointCrossModalLoss(nn.Module):
    """
    Joint loss for cross-modal autoencoder and classification with LfF-style weighting.
    
    Args:
        lambda_self: Weight for self-reconstruction loss
        lambda_cross: Weight for cross-modal reconstruction loss  
        lambda_clf: Weight for classification loss
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
        sample_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute joint loss with optional sample weighting.
        
        Args:
            outputs: Dict from CrossModalAutoencoders.forward()
            fd: Original visual features
            ed: Original text embeddings
            logits: Classification logits
            labels: Ground truth labels
            sample_weights: Optional per-sample weights from LfF (default: None = uniform)
            
        Returns:
            Dict with loss components and total loss
        """
        # Reconstruction losses
        fd_self_loss = self.mse_loss(outputs["fd_recon"], fd).mean(dim=1)
        ed_self_loss = self.mse_loss(outputs["ed_recon"], ed).mean(dim=1)
        ed_cross_loss = self.mse_loss(outputs["ed_recon_from_fl"], ed).mean(dim=1)
        fd_cross_loss = self.mse_loss(outputs["fd_recon_from_el"], fd).mean(dim=1)
        
        self_loss = (fd_self_loss + ed_self_loss).mean()
        cross_loss = (ed_cross_loss + fd_cross_loss).mean()
        recon_loss = self.lambda_self * self_loss + self.lambda_cross * cross_loss
        
        # Classification loss with optional weighting
        clf_loss_per_sample = self.ce_loss(logits, labels)
        
        if sample_weights is not None:
            weighted_clf_loss = (clf_loss_per_sample * sample_weights).mean()
        else:
            weighted_clf_loss = clf_loss_per_sample.mean()
            
        clf_loss = self.lambda_clf * weighted_clf_loss
        
        # Total loss
        total_loss = recon_loss + clf_loss
        
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "self_loss": self_loss,
            "cross_loss": cross_loss,
            "clf_loss": clf_loss,
            "clf_loss_unweighted": clf_loss_per_sample.mean(),
            "fd_self_loss": fd_self_loss.mean(),
            "ed_self_loss": ed_self_loss.mean(),
            "ed_cross_loss": ed_cross_loss.mean(),
            "fd_cross_loss": fd_cross_loss.mean(),
        }


class CrossModalClassifier(nn.Module):
    """
    Trainable classifier head for cross-modal latent features.
    
    Unlike DebiasedClassifier which freezes the autoencoder, this is designed
    for joint training where both autoencoder and classifier are updated.
    
    Args:
        latent_dim: Dimension of latent features (Fl)
        num_classes: Number of output classes
        hidden_dims: Optional hidden layer sizes (default: progressive reduction)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3
    ) -> None:
        super().__init__()
        
        if hidden_dims is None:
            # Create progressive reduction
            if latent_dim >= 256:
                hidden_dims = [latent_dim // 2, latent_dim // 4, max(64, latent_dim // 8)]
            elif latent_dim >= 128:
                hidden_dims = [latent_dim // 2, max(32, latent_dim // 4)]
            else:
                hidden_dims = [max(32, latent_dim // 2)]
        
        layers: List[nn.Module] = []
        prev = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, fl: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fl: Visual latent features [batch_size, latent_dim]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        return self.classifier(fl)


class DebiasedClassifier(nn.Module):
    """
    Classification head using frozen cross-modal autoencoder latent features.

    Args:
        cross_modal_model: Trained CrossModalAutoencoders (will be frozen and eval())
        num_classes: Number of output classes
        dropout: Dropout probability in the classifier head
        hidden_dims: Optional classifier hidden sizes (default: progressive reduction)
    """

    def __init__(
        self,
        cross_modal_model: CrossModalAutoencoders,
        num_classes: int,
        dropout: float = 0.3,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.cross_modal_model = cross_modal_model.eval()
        for p in self.cross_modal_model.parameters():
            p.requires_grad = False

        if hidden_dims is None:
            # Create progressive reduction: latent_dim -> latent_dim//2 -> latent_dim//4 -> num_classes
            latent_dim = self.cross_modal_model.latent_dim
            if latent_dim >= 256:
                hidden_dims = [latent_dim // 2, latent_dim // 4, max(64, latent_dim // 8)]
            elif latent_dim >= 128:
                hidden_dims = [latent_dim // 2, max(32, latent_dim // 4)]
            else:
                hidden_dims = [max(32, latent_dim // 2)]

        in_dim = self.cross_modal_model.latent_dim
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*layers)

    @torch.no_grad()
    def _encode_visual_latent(self, fd: torch.Tensor) -> torch.Tensor:
        return self.cross_modal_model.encoder1(fd)

    def forward(self, fd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fd: Visual features tensor [batch_size, fd_dim]

        Returns:
            Logits tensor [batch_size, num_classes]
        """
        with torch.no_grad():
            fl = self._encode_visual_latent(fd)
        return self.classifier(fl)


def _move_to_device(fd: torch.Tensor, ed: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if fd.device != device:
        fd = fd.to(device)
    if ed.device != device:
        ed = ed.to(device)
    return fd, ed


def train_cross_modal_autoencoders(
    model: CrossModalAutoencoders,
    dataloader,
    num_epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cuda",
    lambda_self: float = 1.0,
    lambda_cross: float = 1.0,
    ckpt_dir: Optional[str] = None,
) -> None:
    """
    Train cross-modal autoencoders with self and cross reconstruction losses.

    Required components:
      - Adam optimizer (weight_decay=1e-5)
      - ReduceLROnPlateau scheduler (patience=10, factor=0.5)
      - Gradient clipping (max_norm=1.0)
      - Logging every 100 batches
      - Checkpoint every 20 epochs with full state

    Args:
        model: CrossModalAutoencoders instance
        dataloader: Iterable yielding (fd, ed) batches
        num_epochs: Number of epochs
        lr: Learning rate
        device: Device string ('cuda' or 'cpu')
        lambda_self: Weight for self reconstruction loss
        lambda_cross: Weight for cross reconstruction loss
        ckpt_dir: Optional directory to save checkpoints; created if missing
    """
    device_t = torch.device(device)
    model.to(device_t)
    criterion = CrossModalLoss(lambda_cross=lambda_cross, lambda_self=lambda_self)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5, verbose=True)

    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_total = 0.0
        running_count = 0

        for batch_idx, batch in enumerate(dataloader):
            # Support dataloaders yielding (fd, ed) or (fd, ed, label)
            fd: torch.Tensor = batch[0]
            ed: torch.Tensor = batch[1]

            fd, ed = _move_to_device(fd, ed, device_t)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(fd, ed)
            loss_dict = criterion(outputs, fd, ed)
            total_loss = loss_dict["total_loss"]

            if torch.isnan(total_loss):
                raise RuntimeError("NaN encountered in total_loss")

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_total += float(total_loss.detach().cpu()) * fd.size(0)
            running_count += int(fd.size(0))

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"[Epoch {epoch:03d}] Batch {batch_idx+1:05d}: "
                    f"total={loss_dict['total_loss']:.4f} self={loss_dict['self_loss']:.4f} cross={loss_dict['cross_loss']:.4f} "
                    f"fd_self={loss_dict['fd_self_loss']:.4f} ed_self={loss_dict['ed_self_loss']:.4f} "
                    f"fd_cross={loss_dict['fd_cross_loss']:.4f} ed_cross={loss_dict['ed_cross_loss']:.4f}"
                )

        epoch_loss = running_total / max(1, running_count)
        scheduler.step(epoch_loss)
        print(f"[Epoch {epoch:03d}] avg_loss={epoch_loss:.6f}")

        if ckpt_dir is not None and (epoch % 20 == 0 or epoch == num_epochs):
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": epoch_loss,
                "hyperparameters": {
                    "fd_dim": model.fd_dim,
                    "ed_dim": model.ed_dim,
                    "latent_dim": model.latent_dim,
                    "hidden_dims": model.hidden_dims,
                },
            }
            path = os.path.join(ckpt_dir, f"cross_modal_epoch_{epoch:03d}.pth")
            torch.save(ckpt, path)
            print(f"Saved checkpoint: {path}")


