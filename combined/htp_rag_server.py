"""
HTP RAG System FastAPI Server

로컬 RAG 시스템을 웹 API로 제공하는 FastAPI 서버
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from sentence_transformers import CrossEncoder
import json
from datetime import datetime
import asyncio

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
sessions = {}  # 세션별 대화 히스토리 관리

# ============================================
# 2. Pydantic 모델 (요청/응답 스키마)
# ============================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    rewritten_queries: List[str]
    source_documents: List[Dict]
    session_id: str

class ResetRequest(BaseModel):
    session_id: Optional[str] = "default"

# ============================================
# 3. 임베딩 래퍼 클래스
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
# 4. 쿼리 재작성기
# ============================================

class AdvancedQueryRewriter:
    def __init__(self, model_name="helena29/Qwen2.5_LoRA_for_HTP"):
        print(f"✅ 쿼리 재작성 모델 로딩 중: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        print(f"✅ 쿼리 재작성 모델 로딩 완료! Device: {self.device}")

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
        if not history_text.strip():
            history_text = "No previous conversation"

        try:
            prompt = self.template.format(
                history_text=history_text,
                current_query=current_query
            )
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant for query rewriting."},
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
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            try:
                import re
                json_match = re.search(r'\{[^{}]*"queries"[^{}]*\}', response, re.DOTALL)
                if json_match:
                    response_json = json.loads(json_match.group())
                else:
                    response_json = json.loads(response)
                
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
# 5. 멀티 쿼리 리트리버
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
# 6. RAG 시스템
# ============================================

class AdvancedConversationalRAG:
    def __init__(self, vectorstore, model_name="helena29/Qwen2.5_LoRA_for_HTP"):
        self.history = []
        self.query_rewriter = AdvancedQueryRewriter(model_name=model_name)
        self.retriever = MultiQueryRetriever(vectorstore=vectorstore, query_rewriter=self.query_rewriter)
        
        print(f"✅ 답변 생성에도 동일 모델 사용: {model_name}")
        self.tokenizer = self.query_rewriter.tokenizer
        self.llm = self.query_rewriter.model
        self.device = self.query_rewriter.device
        print(f"✅ 모델 설정 완료! Device: {self.device}")

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
# 7. 서버 초기화 (시작 시 한 번만 실행)
# ============================================

@app.on_event("startup")
async def startup_event():
    global rag_system
    
    print("=" * 60)
    print("🚀 HTP RAG 서버 시작 중...")
    print("=" * 60)
    
    try:
        # 1. 임베딩 모델 로드
        print("\n[1/3] 임베딩 모델 로드 중...")
        embedding_model_name = "HJUNN/bge-m3b-Art-Therapy-embedding-fine-tuning"
        embed_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        embed_model = AutoModel.from_pretrained(embedding_model_name).to(device)
        embeddings = MyEmbeddings(embed_model, embed_tokenizer, device=device)
        print("✅ 임베딩 모델 로드 완료!")
        
        # 2. 벡터 DB 로드
        print("\n[2/3] 벡터 DB 로드 중...")
        vectorstore = Chroma(
            embedding_function=embeddings,
            collection_name="htp_collection",
            persist_directory="./chroma_store"
        )
        print("✅ 벡터 DB 로드 완료!")
        
        # 3. RAG 시스템 초기화
        print("\n[3/3] RAG 시스템 초기화 중...")
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
# 8. API 엔드포인트
# ============================================

@app.get("/")
async def root():
    """서버 상태 확인"""
    return {
        "status": "running",
        "message": "HTP RAG API Server",
        "device": device,
        "active_sessions": len(sessions)
    }

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

# ============================================
# 9. 서버 실행
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
