from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import config

@lru_cache(maxsize=1)
def get_chat_model() -> ChatGoogleGenerativeAI:

    config.google_key()

    return ChatGoogleGenerativeAI(
        model = config.GEMINI_CHAT_MODEL,
        temperature = config.LLM_TEMPERATURE,
        api_key = config.GOOGLE_API_KEY
    )

@lru_cache(maxsize=1)
def get_embedding_model() -> GoogleGenerativeAIEmbeddings:

    config.google_key()

    return GoogleGenerativeAIEmbeddings(
        model=config.GEMINI_EMBEDDING_MODEL,
        api_key=config.GOOGLE_API_KEY
    )