# HTP 챗봇 로컬 연결 가이드

## 📋 개요
Claude API 대신 로컬 RAG 시스템을 사용하도록 웹 챗봇을 연결했습니다.

## 🔧 시스템 구성

### 1. **백엔드 (FastAPI 서버)**
- **위치**: `c:\Users\helen\Desktop\kt cloud tech up\basic_project\models\combined\htp_rag_server.py`
- **포트**: 8000
- **기능**:
  - HTP 그림 해석
  - 대화형 질문 생성
  - 최종 종합 해석 생성
  - RAG 기반 정보 검색

### 2. **프론트엔드 (React 웹앱)**
- **위치**: `c:\Users\helen\Desktop\kt cloud tech up\basic_project\web\`
- **포트**: 3000
- **기능**:
  - 캔버스 그림 그리기
  - 실시간 채팅 인터페이스
  - 로컬 API 호출

---

## 🚀 실행 방법

### Step 1: Python 환경 확인
```powershell
# 필요한 패키지 확인
pip list | Select-String "fastapi|uvicorn|transformers|langchain|torch|chromadb"
```

필요한 패키지가 없다면:
```powershell
pip install fastapi uvicorn[standard] pydantic transformers langchain-community sentence-transformers chromadb torch
```

### Step 2: 백엔드 서버 실행
**중요**: `models/combined` 폴더에서 실행해야 합니다 (chroma_store 폴더가 있는 곳)

```powershell
# 디렉토리 이동
cd "c:\Users\helen\Desktop\kt cloud tech up\basic_project\models\combined"

# 서버 실행
python htp_rag_server.py
```

또는 uvicorn 직접 사용:
```powershell
uvicorn htp_rag_server:app --reload --host 0.0.0.0 --port 8000
```

**예상 출력**:
```
============================================================
🚀 HTP RAG 서버 시작 중...
============================================================

[1/3] 임베딩 모델 로드 중...
✅ 임베딩 모델 로드 완료!

[2/3] 벡터 DB 로드 중...
✅ 벡터 DB 로드 완료!

[3/3] RAG 시스템 초기화 중...
✅ 쿼리 재작성 모델 로딩 중: helena29/Qwen2.5_LoRA_for_HTP
✅ 쿼리 재작성 모델 로딩 완료! Device: cuda
✅ 답변 생성에도 동일 모델 사용: helena29/Qwen2.5_LoRA_for_HTP
✅ 모델 설정 완료! Device: cuda
✅ RAG 시스템 초기화 완료!

============================================================
✅ 서버 준비 완료!
📍 Device: cuda
🌐 API 문서: http://localhost:8000/docs
============================================================

INFO:     Started server process [...]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 3: 백엔드 테스트
새 PowerShell 창을 열어서:

```powershell
# 서버 상태 확인
Invoke-RestMethod -Uri "http://localhost:8000/" -Method GET

# 테스트 질문
$body = @{
    message = "HTP 검사에서 집을 크게 그린 것은 어떤 의미인가요?"
    session_id = "test"
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Uri "http://localhost:8000/chat" -Method POST -Body $body -ContentType "application/json; charset=utf-8"
```

### Step 4: 프론트엔드 실행
새 PowerShell 창을 열어서:

```powershell
# 웹 디렉토리로 이동
cd "c:\Users\helen\Desktop\kt cloud tech up\basic_project\web"

# 패키지 설치 (처음 한 번만)
npm.cmd install

# 개발 서버 실행
npm.cmd run dev
```

**예상 출력**:
```
VITE v5.0.x  ready in ... ms

➜  Local:   http://localhost:3000/
➜  Network: use --host to expose
```

### Step 5: 브라우저 접속
- 웹 브라우저에서 `http://localhost:3000/` 접속
- 집, 나무, 사람 그림 그리기
- "해석 시작" 버튼 클릭
- 로컬 RAG 시스템이 답변 생성!

---

## 🔍 API 엔드포인트

### 1. 헬스 체크
```
GET http://localhost:8000/
```
응답:
```json
{
  "status": "running",
  "message": "HTP RAG API Server",
  "device": "cuda",
  "active_sessions": 0
}
```

### 2. 채팅
```
POST http://localhost:8000/chat
Content-Type: application/json

{
  "message": "질문 내용",
  "session_id": "user123"
}
```
응답:
```json
{
  "response": "AI 답변",
  "rewritten_queries": ["재작성된 쿼리1", "재작성된 쿼리2"],
  "source_documents": [...],
  "session_id": "user123"
}
```

### 3. 세션 초기화
```
POST http://localhost:8000/reset
Content-Type: application/json

{
  "session_id": "user123"
}
```

### 4. 활성 세션 조회
```
GET http://localhost:8000/sessions
```

### 5. 대화 히스토리 조회
```
GET http://localhost:8000/history/{session_id}
```

### 6. API 문서
```
http://localhost:8000/docs (Swagger UI)
http://localhost:8000/redoc (ReDoc)
```

---

## 🛠️ 문제 해결

### 1. "cannot import name 'Chroma'" 오류
```powershell
pip install --upgrade chromadb langchain-community
```

### 2. "CUDA out of memory" 오류
```python
# htp_rag_server.py에서 device 변경
device = "cpu"  # GPU 대신 CPU 사용
```

### 3. 포트가 이미 사용 중
```powershell
# 8000 포트 사용 중인 프로세스 찾기
netstat -ano | findstr :8000

# 프로세스 종료 (PID 확인 후)
taskkill /PID <PID> /F
```

### 4. CORS 오류
FastAPI 서버가 이미 CORS를 허용하도록 설정되어 있습니다:
```python
allow_origins=["*"]  # 모든 origin 허용
```
프로덕션에서는 specific origin으로 변경 권장.

### 5. 웹에서 "API 호출 실패" 오류
1. 백엔드 서버가 실행 중인지 확인
2. `http://localhost:8000/` 접속 테스트
3. 브라우저 콘솔 (F12) 에서 에러 메시지 확인

---

## 📊 성능 최적화

### GPU 사용 (권장)
- **VRAM 요구사항**: 약 6-8GB
- **속도**: 응답 시간 2-5초
- Qwen2.5-1.5B 모델 (FP16)

### CPU 사용
- **메모리 요구사항**: 약 8-12GB RAM
- **속도**: 응답 시간 10-30초
- `device = "cpu"` 설정

### 모델 로딩 시간
- **첫 실행**: 1-2분 (모델 다운로드)
- **이후 실행**: 30초-1분 (모델 로드)
- 서버 실행 후에는 즉시 응답

---

## 📝 코드 변경 사항

### 변경된 파일:
1. **`web/src/HTPChatbot.tsx`**
   - Claude API 호출 → `callLocalRAG()` 함수로 대체
   - 이미지 base64 전송 제거 (텍스트만 전송)
   - 로컬 API URL: `http://localhost:8000`

### 추가된 파일:
1. **`models/combined/htp_rag_server.py`**
   - FastAPI 서버 구현
   - RAG 시스템 초기화
   - CORS 설정
   - 세션 관리

2. **`models/combined/README_API.md`**
   - API 문서

3. **`models/combined/LOCAL_SETUP_GUIDE.md`** (이 파일)
   - 실행 가이드

---

## 🎯 다음 단계

### 현재 상태:
✅ FastAPI 서버 코드 완성
✅ React 프론트엔드 수정 완료
✅ 로컬 API 호출로 변경
✅ 문서 작성 완료

### 실행 필요:
⏳ 백엔드 서버 실행
⏳ 프론트엔드 서버 실행
⏳ 통합 테스트

### 추가 개선 가능:
- [ ] 이미지 업로드 기능 (base64 → 파일 저장)
- [ ] 스트리밍 응답 (실시간 답변 생성)
- [ ] 세션 지속성 (DB 저장)
- [ ] 로그인/인증 기능
- [ ] 응답 캐싱

---

## 📞 지원

문제가 발생하면:
1. 백엔드 로그 확인
2. 브라우저 콘솔 (F12) 확인
3. `http://localhost:8000/docs` 에서 API 직접 테스트

Happy coding! 🚀
