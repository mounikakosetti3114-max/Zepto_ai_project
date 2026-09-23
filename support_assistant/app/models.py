from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    query: str
    
class SupportResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float