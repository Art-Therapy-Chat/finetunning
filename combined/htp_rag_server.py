"""
HTP RAG System FastAPI Server

로컬 RAG 시스템을 웹 API로 제공하는 FastAPI 서버
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from sentence_transformers import CrossEncoder
import json
from datetime import datetime
import asyncio
import base64
from io import BytesIO
from PIL import Image
import openai
import os

# ============================================
# 1. 설정 및 전역 변수
# ============================================

app = FastAPI(title="HTP RAG API", version="1.0.0")

# CORS 설정 (React 앱에서 접근 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 specific origins으로 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
device = "cuda" if torch.cuda.is_available() else "cpu"
rag_system = None
captioning_model = None
sessions = {}  # 세션별 대화 히스토리 관리

# OpenAI API 키 설정 (환경 변수에서 가져오기)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
openai.api_key = OPENAI_API_KEY

# ============================================
# 2. Pydantic 모델 (요청/응답 스키마)
# ============================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ImageInterpretRequest(BaseModel):
    image: str  # base64 encoded image
    drawing_type: str  # "house", "tree", or "person"

class MultipleImageRequest(BaseModel):
    house: Optional[str] = None  # base64 encoded image
    tree: Optional[str] = None   # base64 encoded image
    person: Optional[str] = None  # base64 encoded image

class ImageInterpretResponse(BaseModel):
    caption: str
    interpretation: str
    rewritten_queries: List[str]
    source_documents: List[Dict]

class MultipleImageResponse(BaseModel):
    house: Optional[ImageInterpretResponse] = None
    tree: Optional[ImageInterpretResponse] = None
    person: Optional[ImageInterpretResponse] = None
    combined_interpretation: str

class ChatResponse(BaseModel):
    response: str
    rewritten_queries: List[str]
    source_documents: List[Dict]
    session_id: str

class ResetRequest(BaseModel):
    session_id: Optional[str] = "default"

# ============================================
# 3. 이미지 캡션 생성 클래스
# ============================================

class ImageCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-large"):
        print(f"✅ 이미지 캡션 모델 로딩 중: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        print(f"✅ 이미지 캡션 모델 로딩 완료! Device: {self.device}")
    
    def generate_caption(self, image_base64: str, drawing_type: str) -> str:
        """
        Base64 인코딩된 이미지에서 캡션 생성
        
        Args:
            image_base64: base64 인코딩된 이미지 문자열
            drawing_type: "house", "tree", "person" 중 하나
        
        Returns:
            str: 생성된 캡션
        """
        try:
            # Base64 디코딩
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data)).convert('RGB')
            
            # 그림 유형에 맞는 프롬프트 설정
            prompts = {
                "house": "A detailed description of this house drawing, including size, windows, doors, chimney, roof, and overall structure:",
                "tree": "A detailed description of this tree drawing, including trunk, branches, leaves, roots, and overall shape:",
                "person": "A detailed description of this person drawing, including body parts, posture, facial features, and overall appearance:"
            }
            
            prompt = prompts.get(drawing_type.lower(), "A description of this drawing:")
            
            # 캡션 생성
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=5,
                    temperature=1.0
                )
            
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            
            # 그림 유형 정보 추가
            full_caption = f"HTP {drawing_type.upper()} drawing: {caption}"
            
            print(f"생성된 캡션 ({drawing_type}): {full_caption}")
            return full_caption
            
        except Exception as e:
            print(f"캡션 생성 실패: {str(e)}")
            return f"HTP {drawing_type} drawing with unclear features"

# ============================================
# 4. 임베딩 래퍼 클래스
# ============================================

class MyEmbeddings(Embeddings):
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            emb = self.model(**inputs).last_hidden_state[:, 0, :]
            emb = emb / emb.norm(dim=1, keepdim=True)
        return emb.cpu().numpy()[0]

# ============================================
# 5. 쿼리 재작성기 (OpenAI API 사용)
# ============================================

class AdvancedQueryRewriter:
    def __init__(self, api_key=None):
        print(f"✅ OpenAI 기반 쿼리 재작성기 초기화")
        self.api_key = api_key or OPENAI_API_KEY
        openai.api_key = self.api_key
        print(f"✅ OpenAI API 설정 완료!")

        self.template = """You are an assistant that regenerates search queries based on the user's previous conversations and questions.

# Instructions
1. Reference all previous queries/retrieved documents/answers in the history below to generate more accurate search queries.
2. If the current question is ambiguous or incomplete, use the history to reconstruct a contextually complete query.
3. If there is no history or it's not relevant, use only the current question.
4. Always generate clear and search-appropriate queries.
5. The output should contain only the regenerated query strings. Do not include additional explanations or comments.
6. If the current sentence contains multiple attributes, separate each into individual queries.
7. Each query should be complete and clear enough to be independently searchable in a vector DB.
8. When combined, the separated queries should represent the meaning of the original query.

# Input
Full conversation history: {history_text}
Current question: {current_query}

# Output Format
You must output in the following JSON format:
{{
    "queries": ["query1", "query2", ...]
}}

Example:
If the current question is "What about Seoul? And restaurants?" and the previous conversation was "Recommend tourist spots in Korea",
{{
    "queries": ["Recommend tourist spots in Seoul", "Recommend restaurants in Seoul"]
}}

Single query case:
{{
    "queries": ["Recommend tourist spots in Korea"]
}}
"""

    def rewrite_query(self, history_text: str, current_query: str) -> List[str]:
        """
        OpenAI API를 사용하여 쿼리 재작성
        """
        if not history_text.strip():
            history_text = "No previous conversation"

        try:
            prompt = self.template.format(
                history_text=history_text,
                current_query=current_query
            )
            
            # OpenAI API 호출
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # 또는 "gpt-3.5-turbo"
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for query rewriting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content
            
            # JSON 파싱
            try:
                import re
                json_match = re.search(r'\{[^{}]*"queries"[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    response_json = json.loads(json_match.group())
                else:
                    response_json = json.loads(response_text)
                
                if "queries" in response_json and isinstance(response_json["queries"], list):
                    return response_json["queries"]
                else:
                    return [current_query]
            except Exception as e:
                print(f"JSON parsing error: {str(e)}")
                return [current_query]
                
        except Exception as e:
            print(f"Error during query rewriting: {str(e)}")
            return [current_query]

# ============================================
# 6. 멀티 쿼리 리트리버
# ============================================

class MultiQueryRetriever:
    def __init__(self, vectorstore, query_rewriter, **kwargs):
        self.vectorstore = vectorstore
        self.query_rewriter = query_rewriter
        self.history = []

    def build_history_text(self) -> str:
        text = ""
        for h in self.history:
            text += f"[QUESTION]\n{h['user_query']}\n"
            text += f"[REWRITTEN QUERIES]\n{h['rewritten_queries']}\n"
            text += "[RETRIEVED DOCS]\n"
            for d in h["retrieved_docs"]:
                text += f"- {d['content']}\n"
            text += f"[ANSWER]\n{h['final_answer']}\n"
            text += "-"*40 + "\n"
        return text

    def retrieve(self, query: str, num_docs=3):
        history_text = self.build_history_text()
        rewritten_queries = self.query_rewriter.rewrite_query(
            history_text=history_text,
            current_query=query
        )

        print(f"원래 쿼리: {query}")
        print(f"재생성된 쿼리들: {rewritten_queries}")

        all_docs = []
        seen_contents = set()

        for idx, rewritten_query in enumerate(rewritten_queries):
            print(f"쿼리 {idx+1} : {rewritten_query}")
            docs = self.vectorstore.similarity_search(rewritten_query, k=num_docs)

            for doc in docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    if not hasattr(doc, "metadata") or doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['query'] = rewritten_query
                    all_docs.append(doc)

        print(f"총 {len(all_docs)}개의 고유 문서를 검색했습니다.")
        return all_docs, rewritten_queries

# ============================================
# 7. RAG 시스템
# ============================================

class AdvancedConversationalRAG:
    def __init__(self, vectorstore, llm_model_name="helena29/Qwen2.5_LoRA_for_HTP"):
        self.history = []
        self.query_rewriter = AdvancedQueryRewriter()  # OpenAI 사용
        self.retriever = MultiQueryRetriever(vectorstore=vectorstore, query_rewriter=self.query_rewriter)
        
        # 답변 생성용 로컬 LLM 로드
        print(f"✅ 답변 생성 모델 로딩 중: {llm_model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            from peft import AutoPeftModelForCausalLM
            self.llm = AutoPeftModelForCausalLM.from_pretrained(
                llm_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
        except:
            base_model = "Qwen/Qwen2.5-1.5B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            self.llm = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            ).to(self.device)
        
        self.llm.eval()
        print(f"✅ 답변 생성 모델 로딩 완료! Device: {self.device}")

        self.response_template = """You are a professional psychologist specialized in HTP (House-Tree-Person) test interpretation.
Your role is to provide clear, professional psychological interpretations based on drawing features.

User Question: {query}

Please provide your interpretation based on the following reference information:
{context}

Guidelines:
1. If the user's question contains multiple queries, address each one clearly and separately.
2. Base your answer only on the provided information. If information is insufficient, honestly state that you don't know.
3. Provide your answer in Korean language.
4. If there are original sources in the provided information, cite them appropriately.
5. Explain possible psychological meanings in a professional manner.

Answer:"""
        
    def generate_response(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a professional psychologist specialized in HTP test interpretation."},
            {"role": "user", "content": prompt}
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
        
    def query(self, current_query: str) -> Dict:
        docs, rewritten_queries = self.retriever.retrieve(current_query)

        if docs:
            context = "\n\n".join([f"문서 {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
            formatted_prompt = self.response_template.format(query=current_query, context=context)
        else:
            formatted_prompt = f"User Question: {current_query}\n\nNo documents were retrieved, but please provide an appropriate answer based on your knowledge."

        response = self.generate_response(formatted_prompt)

        record = {
            "user_query": current_query,
            "rewritten_queries": rewritten_queries,
            "retrieved_docs": [
                {"content": d.page_content, "metadata": d.metadata} for d in docs
            ],
            "final_answer": response
        }
        self.history.append(record)
        self.retriever.history.append(record)

        return {
            "query": current_query,
            "result": response,
            "rewritten_queries": rewritten_queries,
            "source_documents": docs
        }

# ============================================
# 8. 서버 초기화 (시작 시 한 번만 실행)
# ============================================

@app.on_event("startup")
async def startup_event():
    global rag_system, captioning_model
    
    print("=" * 60)
    print("🚀 HTP RAG 서버 시작 중...")
    print("=" * 60)
    
    try:
        # 1. 이미지 캡션 모델 로드
        print("\n[1/4] 이미지 캡션 모델 로드 중...")
        captioning_model = ImageCaptioner()
        print("✅ 이미지 캡션 모델 로드 완료!")
        
        # 2. 임베딩 모델 로드
        print("\n[2/4] 임베딩 모델 로드 중...")
        embedding_model_name = "HJUNN/bge-m3b-Art-Therapy-embedding-fine-tuning"
        embed_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        embed_model = AutoModel.from_pretrained(embedding_model_name).to(device)
        embeddings = MyEmbeddings(embed_model, embed_tokenizer, device=device)
        print("✅ 임베딩 모델 로드 완료!")
        
        # 3. 벡터 DB 로드
        print("\n[3/4] 벡터 DB 로드 중...")
        vectorstore = Chroma(
            embedding_function=embeddings,
            collection_name="htp_collection",
            persist_directory="./chroma_store"
        )
        print("✅ 벡터 DB 로드 완료!")
        
        # 4. RAG 시스템 초기화
        print("\n[4/4] RAG 시스템 초기화 중...")
        rag_system = AdvancedConversationalRAG(vectorstore)
        print("✅ RAG 시스템 초기화 완료!")
        
        print("\n" + "=" * 60)
        print("✅ 서버 준비 완료!")
        print(f"📍 Device: {device}")
        print(f"🌐 API 문서: http://localhost:8000/docs")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 서버 초기화 실패: {str(e)}")
        raise

# ============================================
# 9. API 엔드포인트
# ============================================

@app.get("/")
async def root():
    """서버 상태 확인"""
    return {
        "status": "running",
        "message": "HTP RAG API Server with Image Captioning",
        "device": device,
        "active_sessions": len(sessions),
        "captioning_ready": captioning_model is not None,
        "rag_ready": rag_system is not None
    }

@app.post("/interpret-image", response_model=ImageInterpretResponse)
async def interpret_image(request: ImageInterpretRequest):
    """
    이미지 기반 HTP 해석 엔드포인트
    
    1. 이미지 → 캡션 생성
    2. 캡션 → RAG 검색
    3. 검색 결과 → 해석 생성
    """
    if captioning_model is None:
        raise HTTPException(status_code=503, detail="Captioning model not initialized")
    
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # 1단계: 이미지에서 캡션 생성
        print(f"\n{'='*70}")
        print(f"[1/3] 이미지 캡션 생성 중... (Type: {request.drawing_type})")
        print(f"{'='*70}")
        
        caption = captioning_model.generate_caption(
            request.image,
            request.drawing_type
        )
        
        print(f"생성된 캡션: {caption}")
        
        # 2단계: 캡션 기반으로 RAG 검색 및 해석
        print(f"\n{'='*70}")
        print(f"[2/3] RAG 검색 및 해석 생성 중...")
        print(f"{'='*70}")
        
        result = rag_system.query(caption)
        
        print(f"\n{'='*70}")
        print(f"[3/3] 해석 완료!")
        print(f"{'='*70}")
        print(f"재작성된 쿼리: {result['rewritten_queries']}")
        print(f"검색된 문서 수: {len(result['source_documents'])}")
        print(f"해석 길이: {len(result['result'])} 문자")
        
        return ImageInterpretResponse(
            caption=caption,
            interpretation=result["result"],
            rewritten_queries=result["rewritten_queries"],
            source_documents=[
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        )
        
    except Exception as e:
        print(f"\n❌ 해석 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    채팅 엔드포인트 (일반)
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # 세션별 RAG 시스템 관리
        session_id = request.session_id
        if session_id not in sessions:
            sessions[session_id] = {
                "created_at": datetime.now().isoformat(),
                "message_count": 0
            }
        
        # RAG 쿼리 실행
        result = rag_system.query(request.message)
        
        # 세션 정보 업데이트
        sessions[session_id]["message_count"] += 1
        sessions[session_id]["last_message"] = datetime.now().isoformat()
        
        return ChatResponse(
            response=result["result"],
            rewritten_queries=result["rewritten_queries"],
            source_documents=[
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ],
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/reset")
async def reset_session(request: ResetRequest):
    """
    대화 히스토리 초기화
    """
    session_id = request.session_id
    
    if session_id in sessions:
        del sessions[session_id]
    
    # RAG 시스템 히스토리도 초기화
    if rag_system:
        rag_system.history = []
        rag_system.retriever.history = []
    
    return {
        "message": f"Session {session_id} reset successfully",
        "session_id": session_id
    }

@app.get("/sessions")
async def get_sessions():
    """
    활성 세션 목록 조회
    """
    return {
        "active_sessions": len(sessions),
        "sessions": sessions
    }

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """
    특정 세션의 대화 히스토리 조회
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return {
        "session_id": session_id,
        "history": rag_system.history
    }

@app.post("/interpret-multiple-images", response_model=MultipleImageResponse)
async def interpret_multiple_images(request: MultipleImageRequest):
    """
    멀티 이미지 HTP 해석 엔드포인트
    
    웹에서 3개의 이미지를 한 번에 전송하면 각각 해석 후 종합 해석 제공
    """
    if captioning_model is None:
        raise HTTPException(status_code=503, detail="Captioning model not initialized")
    
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        results = {
            "house": None,
            "tree": None,
            "person": None
        }
        
        all_interpretations = []
        
        print(f"\n{'='*70}")
        print(f"[멀티 이미지 해석 시작]")
        print(f"{'='*70}")
        
        # 각 이미지 타입별로 처리
        for img_type in ["house", "tree", "person"]:
            image_data = getattr(request, img_type)
            
            if image_data:
                print(f"\n[{img_type.upper()}] 처리 중...")
                
                # 1. 캡션 생성
                caption = captioning_model.generate_caption(image_data, img_type)
                print(f"캡션: {caption}")
                
                # 2. RAG 검색 및 해석
                result = rag_system.query(caption)
                
                results[img_type] = ImageInterpretResponse(
                    caption=caption,
                    interpretation=result["result"],
                    rewritten_queries=result["rewritten_queries"],
                    source_documents=[
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in result["source_documents"]
                    ]
                )
                
                all_interpretations.append(f"[{img_type.upper()}]\n{result['result']}")
        
        # 3. 종합 해석 생성
        if all_interpretations:
            combined_prompt = f"""다음은 HTP 검사의 각 그림에 대한 개별 해석입니다:

{chr(10).join(all_interpretations)}

위 개별 해석들을 종합하여 전체적인 심리 상태를 분석해주세요.
- 공통적으로 나타나는 특징
- 세 그림 간의 연관성
- 종합적인 심리 상태 평가
- 긍정적 측면과 발전 방향

한국어로 4-5 문단 정도로 작성해주세요."""
            
            combined_interpretation = rag_system.generate_response(combined_prompt)
        else:
            combined_interpretation = "제공된 이미지가 없어 해석을 생성할 수 없습니다."
        
        print(f"\n{'='*70}")
        print(f"[멀티 이미지 해석 완료]")
        print(f"{'='*70}")
        
        return MultipleImageResponse(
            house=results["house"],
            tree=results["tree"],
            person=results["person"],
            combined_interpretation=combined_interpretation
        )
        
    except Exception as e:
        print(f"\n❌ 멀티 이미지 해석 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing multiple images: {str(e)}")

# ============================================
# 10. 서버 실행
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n🚀 서버 시작...")
    print("📍 주소: http://localhost:8000")
    print("📖 API 문서: http://localhost:8000/docs")
    print("\n종료하려면 Ctrl+C를 누르세요.\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
