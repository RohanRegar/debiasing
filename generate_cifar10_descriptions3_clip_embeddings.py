import json
import torch
from transformers import CLIPProcessor, CLIPModel

def main():
    # Load the model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load the descriptions
    with open("cifar10_descriptions3.json", "r") as f:
        descriptions = json.load(f)
    
    # CIFAR-10 class names in order
    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer", 
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    embeddings = {}
    
    print("Generating CLIP embeddings from cifar10_descriptions3.json...")
    
    for i, class_name in enumerate(cifar10_classes):
        class_data = descriptions[class_name]
        
        # Create comprehensive description combining description and key features
        description = class_data["description"]
        key_features = ", ".join(class_data["key_features"])
        full_description = f"{description} Key features include: {key_features}."
        
        print(f"Class {i} ({class_name}): {full_description}")
        
        # Generate embedding
        inputs = processor(text=[full_description], return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            # Normalize the features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Store embedding as list for JSON serialization
        embeddings[str(i)] = text_features[0].tolist()
    
    # Save embeddings
    output_file = "cifar10_descriptions3_clip_embeddings.json"
    with open(output_file, "w") as f:
        json.dump(embeddings, f)
    
    print(f"\nSaved CLIP embeddings to {output_file}")
    print(f"Embedding dimension: {len(embeddings['0'])}")

if __name__ == "__main__":
    main()

