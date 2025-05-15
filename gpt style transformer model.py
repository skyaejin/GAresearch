import torch
import torch.nn as nn
import math

# Define a Transformer block (decoder block with masked self-attention)
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        # Multi-head self-attention layer
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Feed-forward layers
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        # Layer normalization and dropout
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.ff_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x shape: (batch, seq_len, embed_dim)
        # Self-attention (with residual connection)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)
        x = self.attn_norm(x)
        # Feed-forward (with residual)
        ff_out = self.ff(x)
        ff_out = self.dropout(ff_out)
        x = x + ff_out
        x = self.ff_norm(x)
        return x

# GPT-style Language Model
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=4, max_length=100, dropout=0.1):
        super(GPTLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        # Token embedding and positional embedding
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        # Stack of Transformer decoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim=4*embed_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        # Output projection to vocab size
        self.out_proj = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, input_ids):
        """Forward pass for language model. input_ids: (batch, seq_len) of token indices."""
        batch_size, seq_len = input_ids.size()
        # Create position indices for each token position
        pos_indices = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        # Embed tokens and positions
        tok_embeddings = self.token_emb(input_ids)            # (batch, seq_len, embed_dim)
        pos_embeddings = self.pos_emb(pos_indices)            # (batch, seq_len, embed_dim)
        x = tok_embeddings + pos_embeddings                   # combine token and positional embeddings
        x = self.dropout(x)
        # Create causal mask to mask future tokens: shape (seq_len, seq_len)
        # Mask has True (or -inf) where j > i (future positions), False elsewhere
        device = input_ids.device
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))  # lower triangular matrix
        mask = mask == 0  # convert to boolean mask, True where we should mask (upper triangle)
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        logits = self.out_proj(x)  # (batch, seq_len, vocab_size)
        return logits

# Example: Initialize model and define loss
vocab_size = len(word_to_idx)
gpt_model = GPTLanguageModel(vocab_size=vocab_size, embed_dim=128, num_heads=4, num_layers=4, max_length=100, dropout=0.1)
gpt_model = gpt_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

criterion = nn.CrossEntropyLoss()  # will use ignore_index for padding if needed
optimizer = torch.optim.Adam(gpt_model.parameters(), lr=1e-3)

# Prepare training data for GPT (pad/truncate sequences to max_length and create targets)
pad_token = word_to_idx.get('<PAD>', len(word_to_idx))       # assign a pad token index if not in vocab
eos_token = word_to_idx.get('<EOS>', len(word_to_idx)+1)     # assign an EOS token index if not in vocab
# (If PAD or EOS weren't originally in vocab, we'd add them to vocab and adjust vocab_size accordingly)

# Add PAD and EOS to embeddings if they were new
if pad_token >= vocab_size:
    vocab_size += 1
    gpt_model.token_emb = nn.Embedding(vocab_size, gpt_model.embed_dim).to(gpt_model.token_emb.weight.device)
    gpt_model.out_proj = nn.Linear(gpt_model.embed_dim, vocab_size).to(gpt_model.out_proj.weight.device)
if eos_token >= vocab_size:
    vocab_size += 1
    gpt_model.token_emb = nn.Embedding(vocab_size, gpt_model.embed_dim).to(gpt_model.token_emb.weight.device)
    gpt_model.out_proj = nn.Linear(gpt_model.embed_dim, vocab_size).to(gpt_model.out_proj.weight.device)

# Update word_to_idx and idx_to_word for new tokens if added
word_to_idx['<PAD>'] = pad_token
word_to_idx['<EOS>'] = eos_token
idx_to_word[pad_token] = '<PAD>'
idx_to_word[eos_token] = '<EOS>'

# Pad/truncate token sequences and build input-target pairs
max_len = 100
inputs = []
targets = []
for tokens in news_df['token_ids']:
    # Truncate sequence to max_len-1 to leave space for EOS
    tokens = tokens[:max_len-1]
    tokens.append(eos_token)            # append EOS at end of sequence
    seq_len = len(tokens)
    if seq_len < max_len:
        # Pad sequence if shorter than max_len
        tokens = tokens + [pad_token] * (max_len - seq_len)
    # Create input and target sequences
    # Input is the sequence excluding the last token, target is the sequence excluding the first token
    input_seq = tokens[:-1]   # length = max_len-1
    target_seq = tokens[1:]   # length = max_len-1 (includes EOS as last target for last content token)
    inputs.append(input_seq)
    targets.append(target_seq)
inputs = np.array(inputs)
targets = np.array(targets)

# Convert to torch tensors
inputs_tensor = torch.tensor(inputs, dtype=torch.long)
targets_tensor = torch.tensor(targets, dtype=torch.long)

from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(inputs_tensor, targets_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpt_model.train()
for epoch in range(1, 6):  # let's train for 5 epochs
    total_loss = 0.0
    for batch_inputs, batch_targets in train_loader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()
        # Forward pass
        logits = gpt_model(batch_inputs)  # logits shape: (batch, seq_len, vocab_size)
        # We want to compute loss only on actual tokens (exclude PAD positions)
        # Flatten logits to (batch*seq_len, vocab_size) and targets to (batch*seq_len)
        batch_size, seq_len, vocab_size_ = logits.shape
        loss = criterion(logits.view(-1, vocab_size_), batch_targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} - Average training loss: {avg_loss:.4f}")

# Text generation function using the trained GPT model
def generate_text(model, prompt, max_gen_length=50):
    model.eval()
    tokens = [word_to_idx.get(w, word_to_idx.get('<UNK>', 0)) for w in tokenize_words(normalize_text(prompt))]
    tokens = tokens[:model.max_length-1]  # ensure prompt fits in model context size
    for _ in range(max_gen_length):
        inp = torch.tensor([tokens], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(inp)  # (1, seq_len, vocab_size)
        logits_last = logits[0, -1, :]            # logits for the last generated token
        probas = torch.softmax(logits_last, dim=-1).cpu().numpy()
        next_token = np.random.choice(len(probas), p=probas)  # sample next token (stochastic generation)
        # If you want deterministic, you could do: next_token = probas.argmax()
        if next_token == eos_token:
            break  # end generation if EOS token predicted
        tokens.append(next_token)
        if len(tokens) >= model.max_length:
            break
    # Decode tokens to words
    generated_words = [idx_to_word.get(t, '') for t in tokens]
    return " ".join(generated_words)

# Example generation
prompt = "Breaking news: The government today announced"
print("Prompt:", prompt)
generated_text = generate_text(gpt_model, prompt, max_gen_length=50)
print("Generated article:", generated_text)
