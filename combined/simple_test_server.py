"""
간단한 테스트 서버 - CORS 문제 진단용
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Test Server")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TestRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "서버가 정상 작동 중입니다!"
    }

@app.post("/test")
async def test(request: TestRequest):
    return {
        "received": request.message,
        "response": "메시지를 받았습니다!"
    }

if __name__ == "__main__":
    print("🚀 테스트 서버 시작!")
    print("📍 http://localhost:8000")
    print("Ctrl+C로 종료")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
