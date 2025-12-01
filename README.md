# 🎨 HTP 심리검사 AI 프로젝트

> **HTP (House-Tree-Person)** 그림 심리검사를 위한 AI 기반 이미지 캡셔닝 및 심리학적 해석 시스템

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)

---

## 📋 프로젝트 개요

이 프로젝트는 HTP(집-나무-사람) 그림 심리검사를 AI로 자동화하는 시스템입니다. 
1. **이미지 캡셔닝**: 그림을 텍스트로 설명
2. **심리학적 해석**: 설명을 바탕으로 전문적인 심리 분석 제공
3. **대화형 RAG**: 멀티턴 대화를 통한 심층 상담

---

## 🏗️ 프로젝트 구조

```
models/
├── 📸 captioning/                    # 이미지 캡셔닝 모델들
│   ├── model_blip.ipynb             # BLIP 모델
│   ├── model_llava.ipynb            # LLaVA 모델
│   ├── model_qwen.ipynb             # Qwen-VL 모델 ⭐ 선택
│   └── image_captions_*.json        # 생성된 캡션 결과들
│
├── 🔧 LoRa/                          # LoRA 파인튜닝
│   ├── LoRa.ipynb                   # LoRA 학습 노트북
│   ├── HTP_data.jsonl               # 학습 데이터 (1,453개)
│   └── htp_lora_model/              # 학습된 LoRA 모델
│       ├── adapter_model.safetensors
│       └── adapter_config.json
│
├── ❄️ layer_freezing/                # Layer Freezing 파인튜닝
│   ├── final_htp_model.ipynb        # Layer Freezing 학습
│   ├── test_htp_model.py            # 전체 테스트
│   ├── interactive_test.py          # 대화형 테스트
│   └── qwen2.5-htp-layer-freeze-final/  # 학습된 모델
│       ├── model.safetensors        # 약 3GB
│       ├── config.json
│       └── training_curves.png
│
├── 🤖 combined/                      # RAG 통합 시스템
│   ├── rag_model_combined.ipynb     # 기본 RAG
│   ├── cleaned_멀티턴_멀티쿼리_history_RAG.ipynb  # 멀티턴 대화 RAG
│   ├── chroma_store/                # 벡터 DB
│   └── conversational_rag_history_*.json  # 대화 히스토리
│
├── 📊 test_results/                  # 테스트 결과
│   ├── base_model_test_results.json
│   └── lora_test_results.json
│
├── 📈 Data_generation.ipynb          # 데이터 생성 도구
├── 🧪 test_base_model.ipynb          # 베이스 모델 테스트
├── merge_results.py                  # 결과 병합 스크립트
└── model_comparison_results.csv      # 모델 비교 결과
```

---

## 🚀 핵심 기능

### 1. 🎨 이미지 캡셔닝
HTP 그림을 텍스트로 변환하는 3가지 모델 비교:

| 모델 | 특징 | 성능 |
|------|------|------|
| **BLIP** | 빠르고 가벼움 | 기본적인 설명 생성 |
| **LLaVA** | 상세한 설명 | 중간 수준 |
| **Qwen-VL** ⭐ | 가장 정확하고 상세 | **최종 선택** |

**예시 출력** (Qwen-VL):
```
입력: [나무 그림 이미지]
출력: "The tree is dominant and tall with many branches extending 
       outward. The trunk is thick and sturdy, with deep roots 
       visible at the base."
```

### 2. 🧠 심리학적 해석 (두 가지 방법)

#### A. LoRA 파인튜닝
- **베이스 모델**: Qwen/Qwen2.5-1.5B-Instruct
- **방법**: 어댑터 기반 경량 학습
- **장점**: 빠른 학습, 적은 메모리 (8GB GPU 가능)
- **파라미터**: LoRA rank=8, alpha=32

#### B. Layer Freezing 파인튜닝 ⭐ **추천**
- **베이스 모델**: Qwen/Qwen2.5-1.5B-Instruct
- **방법**: 26/28 레이어 동결, 상위 2개 레이어만 학습
- **장점**: 더 나은 성능, 과적합 방지
- **성능**: Training Loss 0.280 (67% 개선)

**예시 출력**:
```
입력: "The tree is dominant and tall"
AI 해석: "This suggests a strong, assertive personality with a 
          desire for control and leadership. The individual might 
          be perceived as confident and possibly even domineering..."
```

### 3. 💬 대화형 RAG 시스템
멀티턴 대화를 통한 심층 상담:

```
사용자: 나무를 크게 그렸어요
AI: 큰 나무는 자신감과 성장 욕구를 나타냅니다. 가지는 어떻게 그리셨나요?

사용자: 가지가 아래로 처져있어요
AI: 처진 가지는 우울감이나 에너지 저하를 시사할 수 있습니다...
```

**기능**:
- ✅ 대화 히스토리 유지
- ✅ 멀티 쿼리 생성 (5가지 관점)
- ✅ Chroma DB 벡터 검색
- ✅ 컨텍스트 기반 답변

---

## 📊 모델 성능 비교

### 파인튜닝 방법 비교

| 항목 | LoRA | Layer Freezing ⭐ |
|------|------|------------------|
| 학습 파라미터 | 0.79M (0.05%) | ~150M (7-10%) |
| 학습 시간 | ~30분 | ~39분 |
| 최종 Loss | ~0.35 | **0.280** |
| GPU 메모리 | 6GB | 7GB |
| 추론 속도 | 빠름 | 빠름 |
| 품질 | 좋음 | **매우 좋음** |

### 베이스 모델 vs 파인튜닝 비교

```csv
Input: "The tree is large with many branches"

Base Model: 
"The tree symbolizes growth and vitality. [일반적이고 짧은 답변]"

Fine-tuned Model:
"This suggests a strong, assertive personality with a desire for 
control and leadership. The extensive branching indicates a complex 
personality with multiple interests and a tendency to spread oneself 
across various activities. The individual might be perceived as 
confident and possibly even domineering..." [전문적이고 상세한 답변]
```

---

## 🛠️ 설치 및 실행

### 1. 환경 설정

```bash
# Python 3.8+ 필요
pip install torch torchvision transformers
pip install datasets peft bitsandbytes accelerate
pip install chromadb langchain sentence-transformers
pip install pillow matplotlib jupyter
```

### 2. 모델 다운로드

#### LoRA 모델
```bash
# Hugging Face에서 다운로드
git lfs install
git clone https://huggingface.co/helena29/Qwen2.5_LoRA_for_HTP
```

#### Layer Freezing 모델
```bash
# 이미 프로젝트에 포함됨
ls layer_freezing/qwen2.5-htp-layer-freeze-final/
```

### 3. 빠른 시작

#### A. 이미지 캡셔닝
```python
# captioning/model_qwen.ipynb 실행
jupyter notebook captioning/model_qwen.ipynb
```

#### B. 심리학적 해석 (대화형)
```bash
# Layer Freezing 모델 사용
cd layer_freezing
python interactive_test.py
```

#### C. RAG 시스템
```python
# combined/cleaned_멀티턴_멀티쿼리_history_RAG.ipynb 실행
jupyter notebook combined/cleaned_멀티턴_멀티쿼리_history_RAG.ipynb
```

---

## 📝 사용 예시

### 1. Python 스크립트로 사용

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 모델 로드
model_path = "./layer_freezing/qwen2.5-htp-layer-freeze-final"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 프롬프트 생성
prompt = """<|im_start|>system
You are an expert psychologist specialized in HTP test interpretation.<|im_end|>
<|im_start|>user
Please interpret: The tree is very tall with dense branches<|im_end|>
<|im_start|>assistant
"""

# 생성
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.8)
interpretation = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(interpretation)
```

### 2. 대화형 테스트

```bash
cd layer_freezing
python interactive_test.py
```

```
🖼️  이미지 캡션을 입력하세요: The house has no windows

🤖 AI 심리학자가 분석 중...

📋 심리학적 해석:
A house without windows suggests a tendency towards isolation and 
emotional withdrawal. This may indicate difficulty in opening up to 
others or a protective mechanism to avoid vulnerability...

계속하시겠습니까? (y/n):
```

---

## 📁 데이터 구조

### 학습 데이터 형식 (HTP_data.jsonl)

```jsonl
{
  "instruction": "Please provide a psychological interpretation of the following HTP test image caption",
  "input": "The tree is dominant and tall.",
  "output": "This suggests a strong, assertive personality with a desire for control and leadership...",
  "category": "tree"
}
```

- **총 데이터**: 1,453개
- **카테고리**: house, tree, person
- **분할**: 학습 90% (1,307개), 검증 10% (146개)

### 캡션 데이터 형식 (image_captions_qwen.json)

```json
{
  "image_001.jpg": {
    "caption": "The tree is large with many branches...",
    "model": "Qwen-VL",
    "timestamp": "2025-11-15"
  }
}
```

---

## 🔬 연구 및 실험

### 캡셔닝 모델 비교

실험 결과는 `model_comparison_results.csv` 참조:

```bash
python merge_results.py
```

### 코사인 유사도 분석

```bash
# Qwen 임베딩 기반 유사도 계산
cat cosine_similarity_qwen_embedding.csv
```

---

## 🎯 모델 평가

### Layer Freezing 모델 전체 테스트

```bash
cd layer_freezing
python test_htp_model.py
```

**출력**:
- `test_results/test_results_YYYYMMDD_HHMMSS.json` (상세)
- `test_results/test_report_YYYYMMDD_HHMMSS.txt` (요약)

**평가 항목**:
- ✅ 고정 테스트 케이스 (5개)
- ✅ 랜덤 샘플 테스트 (10개)
- ✅ 카테고리별 분석 (house/tree/person)
- ✅ 생성 품질 평가

---

## ⚙️ 하이퍼파라미터

### Layer Freezing 학습 설정

```python
TrainingArguments(
    num_train_epochs=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # 실질 배치: 8
    learning_rate=5e-4,
    warmup_ratio=0.03,
    fp16=True,  # Mixed precision
    gradient_checkpointing=True,
    optim="adamw_torch",
    weight_decay=0.01,
)
```

### 생성 파라미터

```python
generate(
    max_new_tokens=256,
    temperature=0.8,        # 창의성
    top_p=0.95,            # Nucleus sampling
    top_k=40,              # Top-K sampling
    repetition_penalty=1.2, # 반복 방지
    no_repeat_ngram_size=3  # 3-gram 반복 차단
)
```

---

## 🐛 트러블슈팅

### 1. CUDA Out of Memory

```python
# 배치 크기 줄이기
per_device_train_batch_size=1
gradient_accumulation_steps=8

# 또는 max_tokens 줄이기
max_new_tokens=128  # 기본 256
```

### 2. 반복적인 출력 (예: "system system...")

```python
# 이미 해결됨!
repetition_penalty=1.2
no_repeat_ngram_size=3
```

### 3. 모델 로드 실패

```bash
# 경로 확인
ls layer_freezing/qwen2.5-htp-layer-freeze-final/

# 파일 권한 확인 (Linux/Mac)
chmod -R 755 layer_freezing/qwen2.5-htp-layer-freeze-final/
```

### 4. Tokenizer 경고

```python
# pad_token 설정
tokenizer.pad_token = tokenizer.eos_token
```

---

## 📚 기술 스택

| 카테고리 | 기술 |
|----------|------|
| **딥러닝 프레임워크** | PyTorch 2.0+ |
| **모델 라이브러리** | 🤗 Transformers, PEFT |
| **비전 모델** | BLIP, LLaVA, Qwen-VL |
| **언어 모델** | Qwen2.5-1.5B-Instruct |
| **벡터 DB** | ChromaDB |
| **임베딩** | Sentence-Transformers |
| **개발 환경** | Jupyter Notebook, Python 3.8+ |

---

## 🔍 주요 파일 설명

### 학습 관련
- `layer_freezing/final_htp_model.ipynb`: Layer Freezing 학습 전체 과정
- `LoRa/LoRa.ipynb`: LoRA 학습 및 Hugging Face 업로드
- `Data_generation.ipynb`: 학습 데이터 생성 도구

### 추론 및 테스트
- `layer_freezing/test_htp_model.py`: 전체 테스트 스크립트
- `layer_freezing/interactive_test.py`: 대화형 테스트 인터페이스
- `test_base_model.ipynb`: 베이스 모델 성능 테스트

### RAG 시스템
- `combined/rag_model_combined.ipynb`: 기본 RAG 구현
- `combined/cleaned_멀티턴_멀티쿼리_history_RAG.ipynb`: 고급 대화형 RAG

### 유틸리티
- `merge_results.py`: 모델 비교 결과 병합
- `model_comparison_results.csv`: 모델 성능 비교표

---

## 📈 로드맵

### ✅ 완료
- [x] 3가지 캡셔닝 모델 비교 (BLIP, LLaVA, Qwen-VL)
- [x] LoRA 파인튜닝 구현
- [x] Layer Freezing 파인튜닝 구현
- [x] RAG 시스템 통합
- [x] 멀티턴 대화 시스템

### 🚧 진행 중
- [ ] 웹 인터페이스 개발 (Gradio/Streamlit)
- [ ] 더 큰 모델로 확장 (Qwen2.5-7B)
- [ ] 다국어 지원 (한국어 추가)

### 🔮 향후 계획
- [ ] 실시간 그림 분석 API
- [ ] 모바일 앱 통합
- [ ] 전문가 검증 시스템
- [ ] 추가 심리검사 지원 (MBTI, MMPI 등)

---

## 📖 참고 자료

### HTP 검사
- [HTP Test Wikipedia](https://en.wikipedia.org/wiki/House-tree-person_test)
- 투사적 심리검사 기법의 일종
- 무의식적 감정과 성격 특성 분석

### 모델 아키텍처
- [Qwen2.5 Technical Report](https://huggingface.co/Qwen)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Instruction Tuning](https://arxiv.org/abs/2109.01652)

### 관련 논문
- "Scaling Instruction-Finetuned Language Models"
- "Parameter-Efficient Transfer Learning for NLP"

---

## 🤝 기여

이 프로젝트는 KT Cloud Tech Up 프로그램의 일환으로 개발되었습니다.

---

## 📧 문의

문제가 발생하면:
1. GitHub Issues에 등록
2. 로그 파일 첨부
3. 환경 정보 포함 (GPU, Python 버전 등)

---

## 📄 라이센스

이 프로젝트는 연구 및 교육 목적으로 제공됩니다.

---

## 🙏 감사의 말

- **Qwen Team**: 베이스 모델 제공
- **Hugging Face**: 모델 호스팅 및 라이브러리
- **KT Cloud**: 프로젝트 지원

---

**개발 기간**: 2025.11 - 2025.12  
**버전**: 1.0.0  
**최종 업데이트**: 2025-12-01
