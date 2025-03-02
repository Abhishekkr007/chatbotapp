from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chatbot  # This should reference your chatbot module

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_documents(query: QueryRequest):
    response = chatbot.user_input(query.question)
    if not response:
        raise HTTPException(status_code=404, detail="No answer found")
    return {"answer": response}
