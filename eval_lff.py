import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.util import get_dataset, IdxDataset
from module.util import get_model
from util import MultiDimAverageMeter


def evaluate(model, data_loader, target_attr_idx, bias_attr_idx, attr_dims, device):
    """Evaluate model on a given data loader"""
    model.eval()
    attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
    
    for index, data, attr in tqdm(data_loader, desc="Evaluating", leave=False):
        label = attr[:, target_attr_idx]
        data = data.to(device)
        attr = attr.to(device)
        label = label.to(device)
        
        with torch.no_grad():
            logit = model(data)
            pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
            correct = (pred == label).long()

        attr = attr[:, [target_attr_idx, bias_attr_idx]]
        attrwise_acc_meter.add(correct.cpu(), attr.cpu())

    accs = attrwise_acc_meter.get_mean()
    return accs


def main():
    parser = argparse.ArgumentParser(description="Evaluate LFF model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model.th")
    parser.add_argument("--dataset_tag", type=str, required=True, help="Dataset tag")
    parser.add_argument("--model_tag", type=str, default="ResNet20", help="Model architecture")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--target_attr_idx", type=int, default=0, help="Target attribute index")
    parser.add_argument("--bias_attr_idx", type=int, default=1, help="Bias attribute index")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_tag}")
    valid_dataset = get_dataset(
        args.dataset_tag,
        data_dir=args.data_dir,
        dataset_split="eval",
        transform_split="eval",
    )
    
    # Get dataset attributes
    valid_target_attr = valid_dataset.attr[:, args.target_attr_idx]
    valid_bias_attr = valid_dataset.attr[:, args.bias_attr_idx]
    attr_dims = []
    attr_dims.append(torch.max(valid_target_attr).item() + 1)
    attr_dims.append(torch.max(valid_bias_attr).item() + 1)
    num_classes = attr_dims[0]
    
    valid_dataset = IdxDataset(valid_dataset)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = get_model(args.model_tag, num_classes).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Evaluate
    print("Evaluating...")
    accs = evaluate(model, valid_loader, args.target_attr_idx, args.bias_attr_idx, attr_dims, device)
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    overall_acc = torch.mean(accs)
    print(f"Overall Accuracy: {overall_acc:.4f}")
    
    # Calculate aligned and skewed accuracies
    eye_tsr = torch.eye(num_classes)
    aligned_acc = accs[eye_tsr > 0.0].mean()
    skewed_acc = accs[eye_tsr == 0.0].mean()
    
    print(f"Aligned Accuracy (bias-aligned samples): {aligned_acc:.4f}")
    print(f"Skewed Accuracy (bias-conflicting samples): {skewed_acc:.4f}")
    print("\nAttribute-wise Accuracies:")
    print(accs)
    print("="*50)


if __name__ == "__main__":
    main()

