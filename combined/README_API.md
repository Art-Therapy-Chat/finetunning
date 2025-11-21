# HTP RAG API 서버

FastAPI를 사용하여 RAG 시스템을 REST API로 제공합니다.

## 설치 필요한 패키지

```bash
pip install fastapi uvicorn[standard] pydantic
```

## 실행 방법

```bash
python htp_rag_server.py
```

또는

```bash
uvicorn htp_rag_server:app --host 0.0.0.0 --port 8000 --reload
```

## API 엔드포인트

### 1. Health Check
```
GET /
```

### 2. 채팅 (스트리밍 없음)
```
POST /chat
Content-Type: application/json

{
  "message": "손이 크고 눈썹이 진한 사람 그림",
  "session_id": "user123"  // 선택사항
}
```

### 3. 채팅 (스트리밍)
```
POST /chat/stream
Content-Type: application/json

{
  "message": "손이 크고 눈썹이 진한 사람 그림",
  "session_id": "user123"
}
```

### 4. 대화 히스토리 초기화
```
POST /reset
Content-Type: application/json

{
  "session_id": "user123"
}
```

## 테스트

```bash
# 기본 테스트
curl http://localhost:8000/

# 채팅 테스트
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "창문이 큰 집 그림"}'
```
