from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import nest_asyncio
from pyngrok import ngrok
import uvicorn

# Use a pipeline as a high-level helper
from transformers import pipeline

sentiment_model = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")



# Load your trained model (assuming it's a scikit-learn model saved as 'trained_model.pkl')
with open('trained_model.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

# Initialize the FastAPI app
app = FastAPI()

# Define the request body model using Pydantic
class HouseFeatures(BaseModel):
    square_foot: float
    house_age: int
    num_rooms: int

# Define the prediction endpoint
@app.get("/predict")
def predict_house_price(features: HouseFeatures):
    # Extract features from the request body
    input_features = np.array([[features.square_foot, features.house_age, features.num_rooms]])
    # Make a prediction using the loaded model
    prediction = model.predict(input_features)
    # Return the prediction
    return {"predicted_price": prediction[0]}

# Simple root endpoint
@app.get("/")
async def home():
    return "Welcome to the House Price Prediction API"

@app.get("/sentiment/{text}")
async def sentiment(text):
  return str(sentiment_model(text))

# Set up the ngrok tunnel
ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)

# Apply nest_asyncio to run in environments like Jupyter notebooks
nest_asyncio.apply()

# Run the FastAPI app with uvicorn
uvicorn.run(app, port=8000)
