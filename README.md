# Fake News Classifier / Detection


I included two NLP models!

GPT-style News Article Generator: A transformer-based model for generating news articles from a given prompt using a generative pre-trained transformer (GPT) approach.

BERT-based Fake News Classifier: A BERT-based transformer model fine-tuned to classify news articles as fake or real.

**Requirements**

Python 3.x

PyTorch

HuggingFace Transformers library

scikit-learn

NLTK

LIME

ROUGE-score

*install torch trasnformers scikit-learn nltk lime rouge-score   for required dependencies

**Data Preparation**

News Article Data (news_corpus.csv): This CSV file contains column texts with the full text of news articles to train the GPT model for text generation.

Labeled News Data (news_labeled.csv): This CSV file contains column texts with the full text of news articles and a column label with binary labels (0 = real, 1 = fake) to fine-tune the BERT model for fake news detection.



**Data Preprocessing**

The following preprocessing steps are applied to the raw text data:

Normalization: Text is lowercased, and special characters (quotes, slashes, etc.) are removed.

Tokenization:

GPT uses simple whitespace tokenization (you can modify this for more complex tokenization if needed).

BERT uses the pre-trained tokenizer (bert-base-uncased) for tokenizing input.

Padding and Truncation: Texts are padded or truncated to a fixed length (max length = 128 for BERT).



**GPT-style Model for News Article Generation**

The GPT model is a transformer decoder network that learns to predict the next word in a sequence. The key components of this model include:

Token embeddings for each word in the vocabulary.

Positional embeddings to represent the position of words in the sequence.

Transformer decoder blocks for generating text using self-attention.

Output layer for predicting the next word in the sequence.

