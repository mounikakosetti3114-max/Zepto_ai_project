from fastapi import FastAPI
from app.graph import graph
from app.models import (QueryRequest,
                        SupportResponse)

app = FastAPI()

@app.post(
    "/ask",
    response_model = SupportResponse
)
def ask(request:QueryRequest):
    
    result = graph.invoke(
        {
            "query":request.query
        }
    )
    
    return SupportResponse(
        answer = result["answer"],
        
        sources = result["sources"],
        
        confidence = result["confidence"]
    )