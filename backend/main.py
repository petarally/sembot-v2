from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn
from .config import ADMIN_TOKEN
from .router import ChatbotRouter

# Inicijalizacija FastAPI aplikacije
app = FastAPI(title="Fakultetski Chatbot")

# Dodaj CORSMiddleware za pravilno rukovanje preflight OPTIONS zahtjevima
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods, including OPTIONS
    allow_headers=["*"],  # Allow all headers
)

# Inicijalizacija Chatbot routera
chatbot_router = ChatbotRouter()

# Model za zahtjev chatbota
class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)

# Updated response model to support suggested questions
class ChatResponse(BaseModel):
    answer: str
    suggested_questions: list[str] = []

# API endpoint for chatbot
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Endpoint za chatbot upite"""
    response = await chatbot_router.get_response(request.query)
    
    # Check if response is a dict with the new format
    if isinstance(response, dict) and "text" in response:
        return ChatResponse(
            answer=response["text"],
            suggested_questions=response.get("suggested_questions", [])
        )
    
    # Legacy support for string responses
    return ChatResponse(answer=response)

# Health endpoint za Cloud Run / Railway probe
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "qa_pairs": len(chatbot_router.qa_pairs),
        "reranker_ready": chatbot_router.reranker.available,
    }

# --- Admin: punjenje baze bez redeploya -------------------------------------
class AddQaRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    answer: str = Field(min_length=1, max_length=5000)
    paraphrases: list[str] = []


class BulkAddRequest(BaseModel):
    items: list[AddQaRequest] = Field(min_length=1, max_length=2000)


def require_admin(x_admin_token: str = Header(default="")):
    """Propusti samo s ispravnim tokenom. Ako ADMIN_TOKEN nije postavljen, admin je isključen."""
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Admin nije omogućen (ADMIN_TOKEN nije postavljen).")
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Neispravan admin token.")


@app.get("/admin/qa", dependencies=[Depends(require_admin)])
async def list_qa():
    return chatbot_router.list_qa()


@app.post("/admin/qa", dependencies=[Depends(require_admin)])
async def add_qa(req: AddQaRequest):
    try:
        return chatbot_router.add_qa(req.question, req.answer, req.paraphrases)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/admin/qa/bulk", dependencies=[Depends(require_admin)])
async def add_qa_bulk(req: BulkAddRequest):
    """Dodaj više pitanja odjednom (all-or-nothing: ako je ijedan neispravan, ništa se ne sprema)."""
    items = [{"question": i.question, "answer": i.answer, "paraphrases": i.paraphrases} for i in req.items]
    try:
        added = chatbot_router.add_qa_bulk(items)
        return {"added": len(added), "items": added}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/admin/qa/{qa_id}", dependencies=[Depends(require_admin)])
async def delete_qa(qa_id: str):
    if not chatbot_router.delete_qa(qa_id):
        raise HTTPException(status_code=404, detail="Pitanje nije pronađeno.")
    return {"deleted": qa_id}


# Root endpoint za provjeru da API radi
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Fakultetski Chatbot API</title>
        </head>
        <body>
            <h1>Fakultetski Chatbot API</h1>
            <p>API je aktivan. Koristite /api/chat endpoint za interakciju s chatbotom.</p>
        </body>
    </html>
    """

# Ako se pokreće direktno
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)