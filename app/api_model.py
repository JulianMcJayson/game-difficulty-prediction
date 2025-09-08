from pydantic import BaseModel

class GameRequestModel(BaseModel):
    fail : float
    movement_rate : float
    rotation_rate : float
    action_rate : float

class GameResponseModel(BaseModel):
    adaptive_difficulty : int