from fastapi import FastAPI
from app.service_model import predict
from app.api_model import GameRequestModel, GameResponseModel

app = FastAPI()

@app.post("/predict", response_model=GameResponseModel)
async def predict_difficulty(request:GameRequestModel):
    return predict(data=request)