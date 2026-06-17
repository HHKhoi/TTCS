from transformers import pipeline, AutoTokenizer

print("Loading Roberta AI Detector Model (Original Minhkizo)...")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
clf = pipeline(
    "text-classification",
    model="Minhkizo/roberta-ai-detector",
    tokenizer=tokenizer,
    device=-1  # CPU
)
print("Model Loaded Successfully!")

def detect_ai(text):
    if not text or not text.strip():
        return {"label": "Unknown", "score": 0.0}
    try:
        # Using original logic
        result = clf(text[:1000], truncation=True, max_length=512)[0]
        return {
            "label": result["label"],
            "score": result["score"]
        }
    except Exception as e:
        print(f"Detection Error: {e}")
        return {"label": "Error", "score": 0.0}
