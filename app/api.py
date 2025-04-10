from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch, evaluate
rouge_score = evaluate.load("rouge")

t5_model = AutoModelForSeq2SeqLM.from_pretrained("b3nett/t5_news_summarizer")
tokenizer = AutoTokenizer.from_pretrained("t5-small")
bart_model = pipeline("summarization", model="facebook/bart-large-cnn")

def t5_summarize(text, max_input_length=512, max_output_length=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t5_model.to(device)
    inputs = tokenizer(text, return_tensors="pt", max_length=max_input_length, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        summary_ids = t5_model.generate(
            **inputs, 
            max_length=max_output_length, 
            num_beams=5, 
            early_stopping=True
        )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
async def serve_home():
    return "index.html"

class InputData(BaseModel):
    model: str
    data: str

@app.post("/predict")
async def predict(input_data: InputData):
    model = input_data.model
    data = input_data.data

    if model == "bart":
        summary = bart_model(data, max_length=128, min_length=30)
        summary = summary[0]['summary_text']
    elif model == "t5":
        summary = t5_summarize(data)
    else:
        summary = "Please select a model."
    
    return {"prediction": summary}

class RougeData(BaseModel):
    x: str
    y: str

@app.post("/rouge")
async def rouge(rouge_data: RougeData):
    x = rouge_data.x
    y = rouge_data.y
    rouge_scores = rouge_score.compute(predictions=[x], references=[y])
    rounded_rouge_scores = {key: round(value, 2) for key, value in rouge_scores.items()}
    return {"rouge_score": rounded_rouge_scores}