from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
# Jalan ini di terminal 1 : python -m uvicorn api:app --reload
app = FastAPI(
    title="Astra Honda Sentiment API",
    description="API Analisis Sentimen & Topik Otomotif untuk Internal Astra Motor",
    version="2.0.0"
)

MODEL_PATH = "./model_indobert_final"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval() # Set ke mode evaluasi
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

id2label = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_sentiment(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Preprocessing
    inputs = tokenizer(
        request.text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    ).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Hitung probabilitas
        probs = F.softmax(logits, dim=1)
        pred_label_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label_idx].item()

    return {
        "text": request.text,
        "prediction": id2label.get(pred_label_idx, "Unknown"),
        "confidence": round(confidence, 4),
        "label_index": pred_label_idx
    }

@app.get("/")
def root():
    return {"message": "IndoBERT Sentiment API is running!"}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)