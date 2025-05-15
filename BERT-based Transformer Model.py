import torch.nn.functional as F
from transformers import BertModel

# Define a BERT-based classifier model
class BERTFakeNewsClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2):
        super(BERTFakeNewsClassifier, self).__init__()
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        # Classification layer (takes BERT [CLS] hidden state of size 768 for bert-base)
        hidden_size = self.bert.config.hidden_size  # 768 for BERT base
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # BERT model output; we get the pooled output or [CLS] representation
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # `outputs.last_hidden_state` has shape (batch, seq_len, hidden_size)
        # `outputs.pooler_output` is (batch, hidden_size) after a tanh (if using BertModel)
        # We can either use outputs.pooler_output or take last_hidden_state[:,0] which is the [CLS] token embedding
        cls_repr = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_repr)  # shape (batch, num_classes)
        return logits

# Initialize model
bert_classifier = BERTFakeNewsClassifier('bert-base-uncased', num_classes=2)
bert_classifier = bert_classifier.to(device)

# Setup loss and optimizer for fine-tuning
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = torch.optim.AdamW(bert_classifier.parameters(), lr=2e-5)  # low LR for fine-tuning

# Create DataLoader for the classification dataset
train_data = TensorDataset(torch.tensor(X_train_ids, dtype=torch.long),
                           torch.tensor(mask_train, dtype=torch.long),
                           torch.tensor(y_train, dtype=torch.long))
train_loader_cls = DataLoader(train_data, batch_size=16, shuffle=True)
val_data = TensorDataset(torch.tensor(X_val_ids, dtype=torch.long),
                         torch.tensor(mask_val, dtype=torch.long),
                         torch.tensor(y_val, dtype=torch.long))
val_loader_cls = DataLoader(val_data, batch_size=16, shuffle=False)

# Fine-tuning loop
bert_classifier.train()
for epoch in range(1, 4):  # fine-tune for a few epochs
    total_loss = 0.0
    for batch in train_loader_cls:
        batch_input_ids, batch_att_mask, batch_labels = [x.to(device) for x in batch]
        optimizer_cls.zero_grad()
        # Forward pass
        logits = bert_classifier(input_ids=batch_input_ids, attention_mask=batch_att_mask)
        loss = criterion_cls(logits, batch_labels)
        loss.backward()
        optimizer_cls.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader_cls)
    # Optionally evaluate on validation set each epoch
    bert_classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader_cls:
            batch_input_ids, batch_att_mask, batch_labels = [x.to(device) for x in batch]
            outputs = bert_classifier(input_ids=batch_input_ids, attention_mask=batch_att_mask)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
    val_acc = correct / total if total > 0 else 0
    bert_classifier.train()
    print(f"Epoch {epoch} - Training loss: {avg_loss:.4f}, Validation accuracy: {val_acc:.4f}")
