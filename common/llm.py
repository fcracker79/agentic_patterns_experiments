from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from . import keys

def create_openai_llm(k: keys.Keys) -> BaseChatModel:
    llm = ChatOpenAI(temperature=0, api_key=k.openai)
    return llm


def create_ollama_llm(model: str, **kwargs) -> BaseChatModel:
    return ChatOllama(model=model, **kwargs)
