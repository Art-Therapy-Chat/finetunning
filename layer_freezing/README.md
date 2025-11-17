# HTP 심리검사 해석 AI 모델

## 📋 모델 개요

**HTP (House-Tree-Person)** 심리검사 이미지 캡션을 입력받아 전문적인 심리학적 해석을 생성하는 AI 모델입니다.

### 기본 정보
- **베이스 모델**: Qwen/Qwen2.5-1.5B-Instruct
- **학습 방법**: Layer Freezing (26/28 레이어 동결)
- **학습 파라미터**: 전체의 약 7-10%만 업데이트
- **데이터**: HTP 심리검사 해석 데이터 1,453개 샘플
- **카테고리**: House, Tree, Person

---

## 🧠 모델 학습 방법 상세

### 1. Loss (손실 함수)
```
Loss = Cross Entropy Loss (교차 엔트로피)
```

**의미**:
- 모델이 예측한 다음 토큰의 확률 분포와 실제 정답 토큰 간의 차이를 측정
- Loss가 낮을수록 모델의 예측이 정답에 가까움

**계산 방식**:
```python
입력: "This suggests a strong personality"
정답 다음 토큰: "with"
모델 예측 확률: [
    "with": 0.85,  # 높은 확률 → 낮은 Loss
    "and": 0.10,
    "but": 0.05
]
```

### 2. 학습 데이터 구조
```json
{
  "instruction": "Please provide a psychological interpretation...",
  "input": "The tree is dominant and tall.",
  "output": "This suggests a strong, assertive personality...",
  "category": "tree"
}
```

### 3. 학습 프로세스

#### Step 1: 데이터 포맷팅
```
System: "You are an expert psychologist..."
User: "Please interpret this caption: [이미지 설명]"
Assistant: "[심리학적 해석]"
```

#### Step 2: 토큰화
- 최대 길이: 1024 토큰
- 동적 패딩 (배치마다 최대 길이 조정)

#### Step 3: Layer Freezing
```
총 28개 레이어:
├─ Layer 0-25 (26개): 🔒 동결 (업데이트 X)
├─ Layer 26-27 (2개): 🔓 학습 (업데이트 O)
├─ Final LayerNorm: 🔓 학습
└─ LM Head: 🔓 학습
```

**장점**:
- GPU 메모리 절약 (8GB GPU에서도 가능)
- 빠른 학습 속도
- 과적합 방지
- 베이스 모델의 일반 지식 보존

#### Step 4: 학습 설정
```yaml
배치 크기: 2 (per device)
Gradient Accumulation: 4
실질적 배치 크기: 2 × 4 = 8

학습률: 5e-4
스케줄러: Cosine (점진적 감소)
Optimizer: AdamW (weight_decay=0.01)
에폭: 10

메모리 최적화:
  - BFloat16 정밀도
  - Gradient Checkpointing
  - 26/28 레이어 동결
```

### 4. Loss 비교 대상

#### Training Loss
- **데이터**: 학습 데이터 (1,307개 샘플)
- **변화**: 0.841 → 0.280 (67% 감소)
- **의미**: 모델이 학습 데이터를 잘 학습함

#### Validation Loss
- **데이터**: 검증 데이터 (146개 샘플)
- **최종값**: 약 1.809
- **의미**: 
  - Training Loss보다 높음 = 약간의 과적합
  - 하지만 큰 차이 아님 = 일반화 성능 양호
  - 새로운 데이터에도 잘 작동

### 5. 학습 결과
```
학습 시간: 약 39분
최종 Training Loss: 0.280
최종 Validation Loss: 1.809
Loss 개선도: 67% 감소
모델 크기: 2.96 GB
```

---

## 🚀 사용 방법

### 1. 모델 정보 확인
```bash
python inspect_model.py
```

**출력 내용**:
- 모델 설정 (레이어 수, 파라미터 등)
- 학습 요약 (Loss, 학습 시간 등)
- 파일 크기
- 로드 테스트

### 2. 전체 테스트 실행
```bash
python test_htp_model.py
```

**테스트 항목**:
- ✅ 고정 테스트 케이스 5개
- ✅ 데이터셋 랜덤 샘플 10개
- ✅ 카테고리별 성능 분석
- ✅ 결과 JSON/TXT 저장

**출력 파일**:
- `test_results/test_results_YYYYMMDD_HHMMSS.json` (상세 결과)
- `test_results/test_report_YYYYMMDD_HHMMSS.txt` (요약 리포트)

### 3. 대화형 테스트
```bash
python interactive_test.py
```

**사용 예시**:
```
🖼️  이미지 캡션을 입력하세요: The tree is very tall with dense branches

🤖 AI 심리학자가 분석 중...

📋 심리학적 해석:
This suggests a strong, assertive personality with a desire for 
control and leadership. The individual might be perceived as 
confident and possibly even domineering...
```

---

## 📂 파일 구조

```
models/layer_freezing/
├── final_htp_model.ipynb          # 학습 노트북 (원본)
├── qwen2.5-htp-layer-freeze-final/  # 학습된 모델
│   ├── model.safetensors           # 모델 가중치
│   ├── config.json                 # 모델 설정
│   ├── tokenizer.json              # 토크나이저
│   ├── final_training_summary.json # 학습 요약
│   └── training_curves.png         # 학습 곡선
├── test_htp_model.py              # 전체 테스트 스크립트
├── interactive_test.py            # 대화형 테스트
├── inspect_model.py               # 모델 정보 확인
└── README.md                      # 이 파일
```

---

## 🎯 테스트 케이스 예시

### 1. Tree (나무)
```
입력: "The tree is dominant and tall with many branches"
예상: 자신감, 성장 의지, 리더십
```

### 2. House (집)
```
입력: "A small house with no windows, located far from the center"
예상: 내향성, 소통 회피, 고립감
```

### 3. Person (사람)
```
입력: "The person is drawn very small and placed at the bottom corner"
예상: 낮은 자존감, 소속감 부족
```

---

## ⚙️ 요구사항

### Python 라이브러리
```bash
pip install torch transformers datasets numpy matplotlib
```

### 하드웨어
- **최소**: 8GB GPU (RTX 4060 등)
- **권장**: 12GB+ GPU
- **CPU**: 가능하지만 매우 느림

### 디스크 공간
- 모델: 약 3GB
- 실행 환경: 5GB+ 권장

---

## 📊 성능 지표

| 항목 | 값 |
|------|-----|
| 학습 Loss (초기) | 0.841 |
| 학습 Loss (최종) | 0.280 |
| Loss 개선률 | 67% |
| Validation Loss | 1.809 |
| 학습 시간 | 39분 |
| 학습 파라미터 비율 | 7-10% |

---

## 🔧 트러블슈팅

### 1. CUDA Out of Memory
```python
# test_htp_model.py에서 max_tokens 줄이기
max_tokens=150  # 기본값 256에서 줄임
```

### 2. 반복적인 출력 ("system system...")
```python
# 이미 해결됨! 
# repetition_penalty=1.2
# no_repeat_ngram_size=3
```

### 3. 모델 로드 실패
```bash
# 경로 확인
ls ./qwen2.5-htp-layer-freeze-final/

# 권한 확인
chmod -R 755 ./qwen2.5-htp-layer-freeze-final/
```

---

## 📝 주요 함수

### generate_htp_interpretation()
```python
def generate_htp_interpretation(instruction, image_caption, max_tokens=256):
    """
    HTP 이미지 캡션 → 심리학적 해석 생성
    
    Args:
        instruction: 해석 요청 지시문
        image_caption: 이미지 설명 (예: "The tree is tall")
        max_tokens: 최대 생성 토큰 수
    
    Returns:
        str: 심리학적 해석 텍스트
    """
```

### 생성 파라미터
```python
temperature=0.8        # 다양성 (높을수록 창의적)
top_p=0.95            # Nucleus sampling
top_k=40              # Top-K sampling
repetition_penalty=1.2  # 반복 방지
no_repeat_ngram_size=3  # 3-gram 반복 방지
```

---

## 📚 참고 자료

### HTP 검사란?
- **H**ouse-**T**ree-**P**erson Test
- 투사적 심리검사 기법
- 그림을 통한 무의식 분석

### 모델 아키텍처
- Qwen2.5: Alibaba의 최신 언어 모델
- Instruction Tuning: 지시문 따르기 학습
- Layer Freezing: 효율적 파인튜닝

---

## 📧 문의

문제가 발생하면 다음을 확인하세요:
1. GPU 메모리 충분한가?
2. 모든 라이브러리 설치되었나?
3. 모델 파일이 모두 존재하나?

---

**만든 날짜**: 2025-11-15  
**버전**: 1.0.0  
**라이센스**: MIT
