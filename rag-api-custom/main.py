import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassThrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings,ChatOllama

from langchain_postgres import PGVector
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from flashrank import Ranker

load_dotenv()
DB_HOST = os.getenv("DB_HOST", "vectordb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "rm6q65wk")
DB_NAME = os.getenv("DB_NAME", "postgres")

CONNECTION_STRING = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"
COLLECTION_NAME = "librechat_rag_docs"

app=FastAPI(title="Custom RAG Application")

class ChatRequest(BaseModel):
    question:str
    chat_history:List[str]=[]

try:
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",  # Or "e5-small"
        base_url="http://ollama:11434" # Assuming Ollama runs in a separate service called 'ollama'
    )
    vector_store=PGVector(embeddings=embeddings,collection_name=COLLECTION_NAME,connection_string=CONNECTION_STRING)
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
    llm=ChatOllama(model="llama3",temperature=0.3,num_predict=512,keep_alive=-1,num_thread=8)

except Exception as e:
    print(f"Failed to initialize RAG components: {e}")

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
    {"context":retriever | format_docs, "question":RunnablePassThrough()}
    | prompt
    | llm
    |StrOutputParser()
)

@app.post("/ingest")
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
    
@app.post("/chat/completions")
async def chat_completions(request:ChatRequest):
    try:
        response_text=rag_chain.invoke(request.question)
        return {"response":response_text}
    except Exception as e:
        return {"error": str(e), "message": "Failed to process RAG query."}