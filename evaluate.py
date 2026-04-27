import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report


TEXT_COL = "sentences"
LABEL_COL = "labels"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Model yükleniyor...")


model_path = "./final_model"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.to(device)
model.eval()



def predict(text):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return torch.argmax(outputs.logits, dim=1).item()



def evaluate():
    df = pd.read_csv("datasets.csv")

    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        print("HATA: CSV sütunları yanlış")
        print(df.columns.tolist())
        return

    texts = df[TEXT_COL].astype(str).tolist()
    true_labels = df[LABEL_COL].tolist()

    #demo

    texts = texts[:200]
    true_labels = true_labels[:200]


    predictions = []

    print(f"Toplam {len(texts)} veri test ediliyor...\n")

    for i, text in enumerate(texts):
        pred = predict(text)
        predictions.append(pred)

        
        label = "POSITIVE" if pred == 1 else "NEGATIVE"
        print(f"{i+1}. {text[:80]} -> {label}")

    
    acc = accuracy_score(true_labels, predictions)

    report = classification_report(
        true_labels,
        predictions,
        target_names=["NEGATIVE", "POSITIVE"]
    )

    print("\n" + "=" * 50)
    print(" BAŞARI RAPORU")
    print(f"Accuracy: %{acc * 100:.2f}")
    print("\nDetaylı Rapor:")
    print(report)
    print("=" * 50)


if __name__ == "__main__":
    evaluate()

