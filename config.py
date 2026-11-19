from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

def _get_int(name: str, default: int) -> int:

    value = os.getenv(name)
    if value is None or value.strip() == '':
        return default
    try:
        return int(value)
    except ValueError: 
        print(f'{name} is not = {value}')
        return default

# folder
PROJECT_ROOT = Path(__file__).parent

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
FAISS_DIR = VECTORSTORE_DIR / "faiss"

DOC_STORE_PATH = VECTORSTORE_DIR / "doc_store.json"

# model
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
LLM_TEMPERATURE = 0.0


VECTOR_DB = os.getenv("VECTOR_DB", "faiss").strip().lower()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "advanced-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

# chunking
PARENT_CHUNK_SIZE = _get_int("PARENT_CHUNK_SIZE",1600)
PARENT_CHUNK_OVERLAP = _get_int("PARENT_CHUNK_OVERLAP", 200)
CHILD_CHUNK_SIZE = _get_int("CHILD_CHUNK_SIZE", 400)
CHILD_CHUNK_OVERLAP = _get_int("CHILD_CHUNK_OVERLAP", 60)

# retrival pipeline

NUM_QUERY_VARIANTS = _get_int("NUM_QUERY_VARIANTS",3)
VECTOR_TOP_K = _get_int("VECTOR_TOP_K",5)
BM25_TOP_K = _get_int("BM25_TOP_K",6)
HYBRID_TOP_K = _get_int("HYBRID_TOP_K",10)
RERANK_TOP_K = _get_int("RERANK_TOP_K",4)
MIN_RERANK_SCORE = float(os.getenv('MIN_RERANK_SCORE','0.5'))
MAX_CONTEXT_CHARS = _get_int("MAX_CONTEXT_CHARS",8000)
RERANKER = os.getenv("RERANKER",'cross-encoder').strip().lower

CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

MEMORY_WINDOW = _get_int("MEMORY_WINDOW",4)


#checking the environment

def google_key() -> None:
    if not GOOGLE_API_KEY:
        raise ValueError(
            'GOOGLE_API_KEY is not available'
        ) 


def check_pinecone_config() -> None:
    if not PINECONE_API_KEY:
        raise ValueError(
            'key is empty'
        )
    if not PINECONE_INDEX_NAME:
        raise ValueError(
            'index is empty'
        )
    if not PINECONE_NAMESPACE:
        raise ValueError("PINECONE_NAMESPACE is empty")

def describe_item() -> dict:
    return{
        'Chat Model': GEMINI_CHAT_MODEL,
        'Embedding Model': GEMINI_EMBEDDING_MODEL,
        'VectorDB': VECTOR_DB,
        'Parent_chunk_size': PARENT_CHUNK_SIZE,
        'Child_chunk_size': CHILD_CHUNK_SIZE 
    }