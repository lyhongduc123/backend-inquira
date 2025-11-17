from pydantic import BaseModel

class RetrievalRequest(BaseModel):
    query: str