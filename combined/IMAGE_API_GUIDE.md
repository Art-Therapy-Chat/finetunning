# HTP RAG Server - 이미지 해석 API 사용법

## 🎯 주요 변경사항

### 1. **이미지 캡션 생성**
- BLIP 모델 사용하여 HTP 그림에서 특징 추출
- House, Tree, Person 각각에 맞는 프롬프트 적용

### 2. **OpenAI 기반 쿼리 재작성**
- 로컬 모델 대신 OpenAI GPT-4o-mini 사용
- 더 정확한 쿼리 재작성 가능

### 3. **이미지 해석 전용 엔드포인트**
- `/interpret-image`: 이미지 → 캡션 → RAG 검색 → 해석

## 🚀 설정 방법

### 1. OpenAI API 키 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집
OPENAI_API_KEY=sk-your-actual-api-key-here
```

또는 환경 변수로 설정:

```powershell
$env:OPENAI_API_KEY="sk-your-actual-api-key-here"
```

### 2. 서버 실행

```powershell
cd "c:\Users\helen\Desktop\kt cloud tech up\basic_project\models\combined"
python htp_rag_server.py
```

## 📡 API 엔드포인트

### 1. 이미지 해석 (NEW!)

**POST** `/interpret-image`

웹에서 그린 이미지를 base64로 받아서 HTP 해석을 반환합니다.

**요청:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANS...",
  "drawing_type": "house"
}
```

**drawing_type 옵션:**
- `"house"`: 집 그림
- `"tree"`: 나무 그림
- `"person"`: 사람 그림

**응답:**
```json
{
  "caption": "HTP HOUSE drawing: A large house with two windows and a red door...",
  "interpretation": "집 그림의 크기가 크다는 것은...",
  "rewritten_queries": [
    "HTP 검사에서 큰 집의 의미",
    "집 그림에서 창문의 개수와 심리적 의미"
  ],
  "source_documents": [
    {
      "content": "집 그림에서...",
      "metadata": {...}
    }
  ]
}
```

### 2. 서버 상태 확인

**GET** `/`

```json
{
  "status": "running",
  "message": "HTP RAG API Server with Image Captioning",
  "device": "cuda",
  "active_sessions": 0,
  "captioning_ready": true,
  "rag_ready": true
}
```

### 3. 텍스트 채팅 (기존)

**POST** `/chat`

```json
{
  "message": "집을 크게 그린 것은 무슨 의미인가요?",
  "session_id": "user123"
}
```

## 🧪 테스트 방법

### PowerShell에서 테스트:

```powershell
# 이미지를 base64로 인코딩 (예시)
$imageBytes = [System.IO.File]::ReadAllBytes("drawing.png")
$base64Image = [Convert]::ToBase64String($imageBytes)

# API 호출
$body = @{
    image = "data:image/png;base64,$base64Image"
    drawing_type = "house"
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Uri "http://localhost:8000/interpret-image" -Method POST -Body $body -ContentType "application/json; charset=utf-8"
```

### Python에서 테스트:

```python
import requests
import base64

# 이미지 로드
with open("drawing.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# API 호출
response = requests.post(
    "http://localhost:8000/interpret-image",
    json={
        "image": f"data:image/png;base64,{image_base64}",
        "drawing_type": "house"
    }
)

result = response.json()
print("캡션:", result["caption"])
print("해석:", result["interpretation"])
```

## 🔄 처리 흐름

```
웹 (React)
   ↓ base64 이미지 전송
FastAPI Server
   ↓
1️⃣ ImageCaptioner (BLIP)
   → 이미지 분석
   → 특징 추출 (예: "큰 집, 2개 창문, 빨간 문...")
   ↓
2️⃣ AdvancedQueryRewriter (OpenAI)
   → 캡션 기반 쿼리 재작성
   → 예: "HTP 검사에서 큰 집의 의미"
   ↓
3️⃣ MultiQueryRetriever + RAG
   → Chroma DB에서 관련 문서 검색
   → Cross-Encoder로 reranking
   ↓
4️⃣ LLM (Qwen2.5 LoRA)
   → 검색된 문서 기반 해석 생성
   ↓
웹 (React)
   ← 해석 결과 반환
```

## 💰 비용 안내

OpenAI API 사용:
- **gpt-4o-mini**: $0.150 / 1M input tokens, $0.600 / 1M output tokens
- 평균 1회 쿼리 재작성: ~500 tokens (약 $0.0001)
- 매우 저렴합니다!

대안으로 `gpt-3.5-turbo`도 가능 (더 저렴하지만 품질 약간 낮음)

## 🔧 트러블슈팅

### 1. OpenAI API 에러
```
Error: Invalid API key
```
→ `.env` 파일 또는 환경 변수 확인

### 2. CUDA out of memory
- 이미지 캡션 모델(BLIP)이 GPU 메모리 사용
- CPU 모드로 전환: 코드에서 `device = "cpu"` 설정

### 3. 이미지 형식 에러
- Base64 인코딩 확인
- `data:image/png;base64,` 접두사 포함 여부 확인

## 📝 다음 단계

웹 프론트엔드 수정:
```typescript
// HTPChatbot.tsx에서
const interpretDrawing = async (imageBase64: string, type: string) => {
  const response = await fetch('http://localhost:8000/interpret-image', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image: imageBase64,
      drawing_type: type
    })
  });
  
  const result = await response.json();
  return result.interpretation;
};
```
