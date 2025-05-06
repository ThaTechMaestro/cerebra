# inspect_with_transformerlens.py

import torch
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load small TransformerLens-compatible model
model = HookedTransformer.from_pretrained("tiny-stories-1M").to(device)

prompt = "Once upon a time"
tokens = model.to_tokens(prompt).to(device)

# Run with internal caching
logits, cache = model.run_with_cache(tokens)

decoded_output = model.to_string(torch.argmax(logits, dim=-1)[0])
print(f"\nTransformerLens Output: {decoded_output}")

# === Attention Heatmap ===
attn = cache["attn", 0][0, 0].detach().cpu()
tokens_str = model.to_str_tokens(prompt)

plt.figure(figsize=(8, 6))
sns.heatmap(attn, xticklabels=tokens_str, yticklabels=tokens_str, cmap="viridis", annot=True, fmt=".2f")
plt.title("Attention Head 0, Layer 0 (TransformerLens)")
plt.xlabel("Key tokens")
plt.ylabel("Query tokens")
plt.tight_layout()
plt.show()

# === MLP Activation Heatmap ===
mlp_post = cache["post", 0][0].detach().cpu()

plt.figure(figsize=(10, 3))
sns.heatmap(mlp_post.T, cmap="magma", cbar=True)
plt.title("MLP Activations (Layer 0, TransformerLens)")
plt.xlabel("Token Position")
plt.ylabel("Neuron Index")
plt.tight_layout()
plt.show()
