from transformers import pipeline

print("Loading Roberta AI Detector Model...")
clf = pipeline(
    "text-classification",
    model="Minhkizo/roberta-ai-detector",
    device=-1  # CPU
)
print("Model Loaded Successfully!")

def detect_ai(text):
    if not text or not text.strip():
        return {"label": "Unknown", "score": 0.0}
    
    # roberta model max length is 512, but we use 256 here as in original script
    result = clf(text[:256])[0]
    return {
        "label": result["label"],
        "score": result["score"]
    }
