from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from . import keys

def create_openai_llm(k: keys.Keys) -> BaseChatModel:
    llm = ChatOpenAI(temperature=0, api_key=k.openai)
    return llm
