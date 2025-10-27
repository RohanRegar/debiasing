import json
import torch
from transformers import CLIPTokenizer, CLIPTextModel

# CIFAR-10 class names
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
]

# Load CLIP model and tokenizer
model_name = "openai/clip-vit-base-patch16"
tokenizer = CLIPTokenizer.from_pretrained(model_name)
model = CLIPTextModel.from_pretrained(model_name)

embeddings = {}

for class_name in cifar10_classes:
    inputs = tokenizer(class_name, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    embeddings[class_name] = embedding

# Save embeddings to JSON
with open("cifar10_clip_embeddings.json", "w") as f:
    json.dump(embeddings, f)

print("Embeddings saved to cifar10_clip_embeddings.json")
