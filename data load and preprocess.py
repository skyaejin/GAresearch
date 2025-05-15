import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

# Load datasets
news_df = pd.read_csv('news_corpus.csv')       # Contains at least a 'text' column for articles
fake_df = pd.read_csv('news_labeled.csv')      # Contains 'text' and 'label' columns (label: 0 = real, 1 = fake)

# Basic text normalization function
def normalize_text(text):
    text = text.lower()                             # Lowercase
    text = re.sub(r'\s+', ' ', text)                # Collapse multiple spaces/newlines
    text = re.sub(r'[\"\'\\\/]', '', text)          # Remove quotes and slashes (example)
    text = re.sub(r'[^0-9a-zA-Z\s.,;:?!]', ' ', text)  # Remove other special chars, keeping basic punctuation
    return text.strip()

# Apply normalization
news_df['text'] = news_df['text'].apply(normalize_text)
fake_df['text'] = fake_df['text'].apply(normalize_text)

# Tokenization for GPT (simple whitespace tokenization for demonstration)
def tokenize_words(text):
    # Split on whitespace for simplicity; in practice, consider more advanced tokenization (BPE, etc.)
    return text.split()

# Build vocabulary from the news corpus for GPT model
all_tokens = []
for txt in news_df['text']:
    all_tokens.extend(tokenize_words(txt))
# Get unique tokens
vocab = sorted(set(all_tokens))
vocab_size = len(vocab)
word_to_idx = {w: i for i, w in enumerate(vocab, start=0)}  # map word to integer ID
idx_to_word = {i: w for w, i in word_to_idx.items()}

# Encode news articles to sequences of token IDs
news_df['token_ids'] = news_df['text'].apply(lambda txt: [word_to_idx[w] for w in tokenize_words(txt) if w in word_to_idx])

# Prepare data for classification model using a BERT tokenizer (WordPiece)
!pip install transformers  # install the transformers library for tokenizer (if not already installed)
from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and encode text for BERT (adding [CLS] and [SEP] tokens, padding to a max length)
max_length = 128  # maximum sequence length for BERT input
texts = fake_df['text'].tolist()
labels = fake_df['label'].tolist()

input_ids = []
attention_masks = []
for txt in texts:
    encoded = bert_tokenizer.encode_plus(
        txt, 
        add_special_tokens=True,        # adds [CLS] and [SEP]
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_attention_mask=True
    )
    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])
input_ids = np.array(input_ids)
attention_masks = np.array(attention_masks)
labels = np.array(labels)

# Split the classification dataset into train/val/test
X_train_ids, X_temp_ids, y_train, y_temp, mask_train, mask_temp = train_test_split(
    input_ids, labels, attention_masks, test_size=0.2, random_state=42)
X_val_ids, X_test_ids, y_val, y_test, mask_val, mask_test = train_test_split(
    X_temp_ids, y_temp, mask_temp, test_size=0.5, random_state=42)

print("GPT vocab size:", vocab_size)
print("Classification dataset training examples:", len(X_train_ids))
