from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
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
    query: str

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