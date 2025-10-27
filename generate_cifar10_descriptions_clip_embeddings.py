import json
import torch
from transformers import CLIPTokenizer, CLIPTextModel

# Load descriptions
with open("cifar10_descriptions2.json", "r") as f:
    descriptions = json.load(f)

model_name = "openai/clip-vit-base-patch16"
tokenizer = CLIPTokenizer.from_pretrained(model_name)
model = CLIPTextModel.from_pretrained(model_name)

output_dict = {}

for class_name, class_data in descriptions.items():
    # Create a comprehensive description combining multiple fields
    one_line = class_data["one_line_caption"]
    shape_parts = class_data["typical_shape_and_parts"]
    colors_texture = class_data["typical_colors_and_texture"]
    distinctive_features = ", ".join(class_data["distinctive_features_to_disambiguate"])
    
    # Combine into a rich description
    description = f"{one_line} {shape_parts} It typically has {colors_texture}. Key features: {distinctive_features}."
    
    inputs = tokenizer(description, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    output_dict[class_name] = {
        "description": description,
        "embedding": embedding
    }

with open("cifar10_descriptions_clip_embeddings.json", "w") as f:
    json.dump(output_dict, f)

print("Embeddings with descriptions saved to cifar10_descriptions_clip_embeddings.json")
