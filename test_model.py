from transformers import pipeline

# ÖNEMLİ: Burası 'train.py' içindeki save_model yolu ile aynı olmalı!
model_path = "./final_model" 

# Pipeline'ı yükle
# Burada tokenizer olarak yine model_path'i veriyoruz ki 
# eğitilmiş (fine-tuned) özel tokenları doğru okusun.
classifier = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

# Test edilecek cümle
#text = "This was the worst movie I have ever seen, a total waste of time."
text = "The movie was absolutely fantastic, ı love every second of it."

# Tahmin al
result = classifier(text)

print(f"Yorum: {text}")
print(f"Sonuç: {result}")