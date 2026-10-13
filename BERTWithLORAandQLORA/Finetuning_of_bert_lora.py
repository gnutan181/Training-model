from datasets import load_dataset
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
      Trainer, 
    TrainingArguments,
    pipeline) 
from sklearn.metrics import accuracy_score, f1_score
# ─────────────────────────────────────────────────────────────────────────────
# 1. Load the dataset
# ─────────────────────────────────────────────────────────────────────────────

dataset = load_dataset('dair-ai/emotion')

df = dataset['train'].to_pandas()

label_names = dataset['train'].features['label'].names
df['label_text'] = df['label'].apply(lambda x : label_names[x])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tokenization
# ─────────────────────────────────────────────────────────────────────────────
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Train / Test/ Validation Split (70/20/10)
# ─────────────────────────────────────────────────────────────────────────────
#70% of data = Train data, 20% of data = Test Data, 10% of data = Validation data
train, test = train_test_split(df, test_size=0.3, stratify=df['label_text'])
test, validation = train_test_split(test, test_size=1/3, stratify=test['label'])


# ─────────────────────────────────────────────────────────────────────────────
# 4.Building a hugging face DatasetDict
# ─────────────────────────────────────────────────────────────────────────────

dataset = DatasetDict({
    'train' : Dataset.from_pandas(train,preserve_index=False),
    'test' : Dataset.from_pandas(test,preserve_index=False),
    'validation' : Dataset.from_pandas(validation,preserve_index=False)
})


# ─────────────────────────────────────────────────────────────────────────────
# 5.Tokenize all Splits via dataset.map
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(batch):
    return tokenizer(batch['text'],padding =True,truncation=True)


encoded = dataset.map(tokenize,batched=True,batch_size =None)


# ─────────────────────────────────────────────────────────────────────────────
# 6.Building the Model
# ─────────────────────────────────────────────────────────────────────────────
label2id = {label: i for i, label in enumerate(label_names)}
id2label = {i: label for i, label in enumerate(label_names)}


if torch.has_mps:
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    

config = AutoConfig.from_pretrained(model_checkpoint, label2id=label2id, id2label=id2label)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 7.Training Args
# ─────────────────────────────────────────────────────────────────────────────
batch_size = 64
training_dir = "bert_base_uncased_trained__lora_model"

training_args = TrainingArguments(output_dir=training_dir,
                                  num_train_epochs= 2,
                                  learning_rate= 2e-5,
                                  per_device_eval_batch_size=batch_size,
                                  per_device_train_batch_size=batch_size,
                                  weight_decay=0.01,
                                  eval_strategy='epoch',
                                  disable_tqdm=False)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Compute Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    
    return {"accuracy": acc, "F1 Score": f1}


# ─────────────────────────────────────────────────────────────────────────────
# 9. Trainer
# ─────────────────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset= encoded['train'],
    eval_dataset=encoded['validation'],
    processing_class=tokenizer
)

# ─────────────────────────────────────────────────────────────────────────────
# 10. Training
# ─────────────────────────────────────────────────────────────────────────────
trainer.train()
