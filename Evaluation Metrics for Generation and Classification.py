#pip install nltk rouge-score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Example evaluation on one prompt from test set
test_prompt = "Scientists discovered a new species of dinosaur"
reference_continuation = "scientists discovered a new species of dinosaur in argentina, shedding light on the jurassic era."  # example reference

generated_continuation = generate_text(gpt_model, test_prompt, max_gen_length=30)
print("Generated:", generated_continuation)
print("Reference:", reference_continuation)

# Compute BLEU score (e.g., using 4-gram BLEU)
reference_tokens = tokenize_words(normalize_text(reference_continuation))
generated_tokens = tokenize_words(normalize_text(generated_continuation))
bleu_score = sentence_bleu([reference_tokens], generated_tokens)  # single reference BLEU
print(f"BLEU score: {bleu_score:.4f}")

# Compute ROUGE scores (e.g., ROUGE-1, ROUGE-2, ROUGE-L)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference_continuation, generated_continuation)
print("ROUGE-1 (precision/recall/f1):", scores['rouge1'])
print("ROUGE-2 (precision/recall/f1):", scores['rouge2'])
print("ROUGE-L (precision/recall/f1):", scores['rougeL'])

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Prepare test DataLoader
test_data = TensorDataset(torch.tensor(X_test_ids, dtype=torch.long),
                          torch.tensor(mask_test, dtype=torch.long),
                          torch.tensor(y_test, dtype=torch.long))
test_loader_cls = DataLoader(test_data, batch_size=16, shuffle=False)

bert_classifier.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_loader_cls:
        batch_input_ids, batch_att_mask, batch_labels = [x.to(device) for x in batch]
        outputs = bert_classifier(input_ids=batch_input_ids, attention_mask=batch_att_mask)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(list(preds))
        all_labels.extend(list(batch_labels.cpu().numpy()))

# Compute metrics
acc = accuracy_score(all_labels, all_preds)
prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
print(f"Test Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}, Recall: {rec:.3f}, F1-Score: {f1:.3f}")


!pip install lime
from lime.lime_text import LimeTextExplainer

# Instantiate LIME text explainer
class_names = ['Real', 'Fake']
explainer = LimeTextExplainer(class_names=class_names)

# Define a prediction function for LIME that takes a list of texts and returns prediction probabilities
def predict_proba(text_list):
    bert_classifier.eval()
    inputs = bert_tokenizer(text_list, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_classifier(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        probs = F.softmax(outputs, dim=1).cpu().numpy()
    return probs

# Pick a test example to explain
idx = 0
sample_text = fake_df['text'].iloc[len(X_train_ids)+len(X_val_ids)+idx]  # first test sample text
sample_label = fake_df['label'].iloc[len(X_train_ids)+len(X_val_ids)+idx]
print("Truth:", "Fake" if sample_label==1 else "Real")
print("Text:", sample_text[:200], "...")  # print first 200 chars

# Get LIME explanation for this sample
exp = explainer.explain_instance(sample_text, predict_proba, num_features=10, labels=(0,1))
# The explain_instance returns explanation for each label; let's get explanation for the predicted label
pred_label = int(bert_classifier(torch.tensor([input_ids[len(X_train_ids)+len(X_val_ids)+idx]], device=device), 
                                 attention_mask=torch.tensor([attention_masks[len(X_train_ids)+len(X_val_ids)+idx]], device=device)
                                ).argmax().cpu().item())
print("Predicted label:", class_names[pred_label])
# Get explanation weights for the predicted label
exp_list = exp.as_list(label=pred_label)
print(f"LIME explanation for label {class_names[pred_label]}:")
for feature, weight in exp_list:
    print(f"  {feature} => weight {weight:.3f}")
