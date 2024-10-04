from pydantic import BaseModel

class StateErrorMessage(BaseModel):
    message: str

