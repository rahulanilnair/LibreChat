import os
from dotenv import load_dotenv
from fastapi import FastAPI,Request
from pydantic import BaseModel
from typing import List
from fastapi.responses import StreamingResponse
import json
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings,ChatOllama

from langchain_postgres import PGVector
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from flashrank import Ranker
from sqlalchemy import create_engine

load_dotenv()
DB_HOST = os.getenv("DB_HOST", "vectordb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "postgres")

CONNECTION_STRING = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"
COLLECTION_NAME = "librechat_rag_docs"

app=FastAPI(title="Custom RAG Application")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # This prints EVERY request hitting the container
    print(f"👀 INCOMING REQUEST: {request.method} {request.url.path}")
    response = await call_next(request)
    return response

class Message(BaseModel):
    role: str
    content: str

class OpenAIRequest(BaseModel):
    model: str = "llama3.2" 
    messages: List[Message] 
    temperature: float = 0.3
    stream: bool = False


embeddings = OllamaEmbeddings(
    model="nomic-embed-text",  
    base_url="http://ollama:11434" 
)


@app.get("/v1/health")
async def health_check():
    return {"status": "ok"}

engine = create_engine(CONNECTION_STRING)
vector_store = PGVector(embeddings=embeddings, collection_name=COLLECTION_NAME, connection=engine)
base_retriever=vector_store.as_retriever(search_kwargs={"k":15})
reranker_client = Ranker(model_name="ms-marco-MiniLM-L-12-v2", max_length=512)
compressor = FlashrankRerank(
    client=reranker_client, 
    top_n=4 
)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
llm = ChatOllama(
    model="llama3.2",
    base_url="http://ollama:11434", # <--- CRITICAL FIX
    temperature=0.3,
    num_predict=512,
    keep_alive=-1,
    num_thread=8
)
template="""
You are an expert RAG assistant. Use the following context to answer the user's question. 
If the answer is not in the context, state that you cannot answer based on the provided documents.

CONTEXT:
{context}

QUESTION:
{question}
"""

prompt=ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

rag_chain=(
    {"context":retriever | format_docs, "question":RunnableLambda(lambda x: x)}
    | prompt
    | llm
    |StrOutputParser()
)


@app.post("/v1/ingest")
async def ingest_documents(file_path: str):
    try:
        loader=TextLoader(file_path)
        documents=loader.load()
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=100)
        chunks=text_splitter.split_documents(documents)
        vector_store.add_documents(chunks)
        return {"status": "success", "message": f"Successfully indexed {len(chunks)} chunks."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    

    
@app.post("/v1/chat/completions")
async def chat_completions(request: OpenAIRequest):
    global rag_chain
    async def generate_stream():
        last_user_message=request.messages[-1].content
        print(f"📩 Received (Streaming): {last_user_message}")
        for chunk in rag_chain.stream(last_user_message):
            data = {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 1677652288,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0)
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate_stream(), media_type="text/event-stream")