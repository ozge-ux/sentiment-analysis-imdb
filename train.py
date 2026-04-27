import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ======================
# 1. GPU AYARI
# ======================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))



print("Loading dataset...")
dataset = load_dataset("imdb")

# shuffle (önemli: overfitting azaltır)
dataset = dataset.shuffle(seed=42)



checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128   
    )


print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_fn, batched=True)



model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1}
)



training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,

    evaluation_strategy="epoch",
    save_strategy="epoch",

    logging_steps=200,
    fp16=True,

    report_to="none"  
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)



print("Training started...")
trainer.train()



trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")

print("DONE ✔ Model saved to ./final_model")
