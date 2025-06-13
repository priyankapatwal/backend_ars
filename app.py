from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel, constr
from transformers import pipeline

# Initialize the FastAPI app
app = FastAPI()

# Add the CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","*"],  # You can change this to specific origins in production, like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers (authorization, content-type, etc.)
)

# Load the multilingual sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Define the input model
class TextInput(BaseModel):
    text: constr(min_length=1)  # Ensure text is not empty

@app.post("/api/analyze")
async def analyze(text_input: TextInput):
    # The pipeline can handle multiple languages, so we can use the same one for any language
    result = sentiment_pipeline(text_input.text)[0]  # Get the first result
    label = result['label']  # e.g., "POSITIVE" or "NEGATIVE"
    score = result['score']  # Probability score
    stars = int(label.split()[0])
    if stars <= 2:
        sentiment = "NEGATIVE"
    elif stars == 3:
        sentiment = "NEUTRAL"
    else:
        sentiment = "POSITIVE"

    return {
        "sentiment": sentiment,
        "score": round(score, 4),
        "feedback": "Text analyzed successfully."
    }

@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API"}