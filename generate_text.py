import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained model and tokenizer
model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Input text to generate from
input_text = "Once upon a time"
print(f"Input prompt: {input_text}")

# Tokenize input text
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Generate text
attention_mask = torch.ones_like(input_ids)  # Create explicit attention mask
output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=50,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    do_sample=True,  # Enable sampling for temperature, top_k and top_p to take effect
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated text:")
print(generated_text)