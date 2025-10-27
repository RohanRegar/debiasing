import json
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _to_tensor_float32(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.float()


def load_clip_embeddings_from_json(
    json_path: str,
    label_names: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """
    Load CLIP text embeddings from a JSON file and return a tensor of shape [C, D].

    Supports the following JSON formats:
      1) {"0": [...], "1": [...], ...}  # keyed by class index (as string)
      2) {"airplane": [...], ...}        # keyed by class name
      3) [[...], [...], ...]              # list ordered by class index

    Args:
        json_path: Path to JSON file containing embeddings
        label_names: Optional list of class names to align name-keyed dicts

    Returns:
        Tensor of shape [num_classes, embedding_dim]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        arr = np.asarray(data, dtype=np.float32)
        return torch.from_numpy(arr)

    if isinstance(data, dict):
        # Case 1: stringified indices
        if all(k.isdigit() for k in data.keys()):
            indices = sorted(int(k) for k in data.keys())
            arr = np.asarray([data[str(i)] for i in indices], dtype=np.float32)
            return torch.from_numpy(arr)

        # Case 2: class-name keyed
        if label_names is None:
            raise ValueError(
                "label_names is required to align name-keyed embeddings (e.g., CIFAR-10 class names)."
            )
        matrix: List[List[float]] = []
        for name in label_names:
            if name not in data:
                raise KeyError(f"Missing embedding for class name: {name}")
            val = data[name]
            # Support objects like {"description": str, "embedding": [..]}
            if isinstance(val, dict):
                if "embedding" not in val:
                    raise KeyError(f"Class '{name}' entry missing 'embedding' key")
                vec = val["embedding"]
            else:
                vec = val
            matrix.append(vec)
        arr = np.asarray(matrix, dtype=np.float32)
        return torch.from_numpy(arr)

    raise TypeError("Unsupported JSON structure for embeddings. Use list or dict.")


class CrossModalPairedDataset(Dataset):
    """
    Dataset yielding (fd, ed) pairs aligned by class label.

    Expects precomputed visual features (fd) and corresponding integer labels.
    Text embeddings are provided as a [num_classes, ed_dim] matrix and are indexed
    by the sample's class label to produce ed.

    Args:
        fd: Feature array or tensor of shape [N, fd_dim] or path to .npy file
        labels: Label array of shape [N] or path to .npy file
        ed_embeddings: Tensor of shape [num_classes, ed_dim]
    """

    def __init__(
        self,
        fd: Union[str, np.ndarray, torch.Tensor],
        labels: Union[str, np.ndarray, torch.Tensor],
        ed_embeddings: torch.Tensor,
    ) -> None:
        super().__init__()

        if isinstance(fd, str):
            if not os.path.exists(fd):
                raise FileNotFoundError(fd)
            fd_arr = np.load(fd)
        else:
            fd_arr = fd

        if isinstance(labels, str):
            if not os.path.exists(labels):
                raise FileNotFoundError(labels)
            labels_arr = np.load(labels)
        else:
            labels_arr = labels

        self.fd = _to_tensor_float32(fd_arr)
        self.labels = torch.as_tensor(labels_arr, dtype=torch.long)
        self.ed_embeddings = ed_embeddings.float()

        if self.fd.ndim != 2:
            raise ValueError("fd must be 2D: [N, fd_dim]")
        if self.labels.ndim != 1:
            raise ValueError("labels must be 1D: [N]")
        if self.fd.size(0) != self.labels.size(0):
            raise ValueError("fd and labels must have matching first dimension")

    def __len__(self) -> int:
        return self.fd.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fd = self.fd[idx]
        label_t = self.labels[idx]
        label = int(label_t.item())
        ed = self.ed_embeddings[label]
        return fd, ed, label_t


def build_dataloader_from_paths(
    fd_npy_path: str,
    labels_npy_path: str,
    embeddings_json_path: str,
    label_names: Optional[Sequence[str]],
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Convenience to assemble a DataLoader from numpy paths and an embeddings JSON.
    """
    ed_matrix = load_clip_embeddings_from_json(embeddings_json_path, label_names)
    ds = CrossModalPairedDataset(fd_npy_path, labels_npy_path, ed_matrix)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def create_cross_modal_dataloader(
    fd_data: np.ndarray,
    labels: np.ndarray,
    clip_embeddings: Dict[int, List[float]],
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader from numpy arrays and CLIP embeddings dictionary.
    
    Args:
        fd_data: Visual features array of shape [N, fd_dim]
        labels: Labels array of shape [N]
        clip_embeddings: Dictionary mapping class indices to CLIP embeddings
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader for cross-modal training
    """
    from torch.utils.data import TensorDataset
    
    # Convert CLIP embeddings dict to tensor matrix
    num_classes = len(clip_embeddings)
    embedding_dim = len(next(iter(clip_embeddings.values())))
    ed_matrix = torch.zeros(num_classes, embedding_dim)
    
    for class_idx, embedding in clip_embeddings.items():
        ed_matrix[class_idx] = torch.tensor(embedding, dtype=torch.float32)
    
    # Create tensors
    fd_tensor = torch.from_numpy(fd_data).float()
    labels_tensor = torch.from_numpy(labels).long()
    
    # Get embeddings for each sample based on labels
    ed_tensor = ed_matrix[labels_tensor]
    
    # Create dataset and dataloader
    dataset = TensorDataset(fd_tensor, ed_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


class LfFCrossModalDataset(torch.utils.data.Dataset):
    """
    Dataset for LfF-weighted cross-modal training.
    
    Returns original images paired with text embeddings and indices for EMA tracking.
    
    Args:
        base_dataset: Dataset from get_dataset() that returns (image, attr)
        clip_embeddings: Dict mapping class indices to text embeddings
        target_attr_idx: Index of target attribute in attr tensor (default: 0)
    """
    
    def __init__(
        self,
        base_dataset,
        clip_embeddings: Dict[int, torch.Tensor],
        target_attr_idx: int = 0
    ):
        self.base_dataset = base_dataset
        self.clip_embeddings = clip_embeddings
        self.target_attr_idx = target_attr_idx
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get image and attributes from base dataset
        if hasattr(self.base_dataset, '__getitem__'):
            item = self.base_dataset[idx]
            if len(item) == 3:  # IdxDataset returns (idx, image, attr)
                _, image, attr = item
                sample_idx = idx
            else:  # Regular dataset returns (image, attr)
                image, attr = item
                sample_idx = idx
        else:
            image, attr = self.base_dataset[idx]
            sample_idx = idx
            
        # Extract target label
        label = attr[self.target_attr_idx].item() if torch.is_tensor(attr[self.target_attr_idx]) else attr[self.target_attr_idx]
        
        # Get corresponding text embedding
        text_embedding = self.clip_embeddings[label]
        
        return image, text_embedding, label, sample_idx


