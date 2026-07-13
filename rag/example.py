import typing
from functools import partial
from typing import TypedDict

import weaviate
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSerializable
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from common.keys import get_keys
from common.llm import create_openai_llm


def _get_retriever():
    loader = TextLoader('./state_of_the_union.txt')
    documents = loader.load()
    # Chunk documents
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    # Embed and store chunks in Weaviate
    client = weaviate.connect_to_embedded()
    vectorstore = WeaviateVectorStore.from_documents(
        client=client,
        documents=chunks,
        embedding=OpenAIEmbeddings(api_key=get_keys().openai),
    )
    # Define the retriever
    retriever = vectorstore.as_retriever()
    return retriever


class RAGGraphState(TypedDict):
    question: str
    documents: typing.List[Document]
    generation: str


def _retrieve_documents_node(state: RAGGraphState, retriever: RunnableSerializable[str, typing.List[Document]], llm: BaseChatModel) -> RAGGraphState:
    """Retrieves documents based on the user's question."""
    question = state["question"]
    documents = retriever.invoke(question)

    prompt = ChatPromptTemplate.from_messages([
        (
            "human",
            """
            Given question {question}, do the following documents provide relevant context?
            Respond with just YES or NO.
            """
        ),
        MessagesPlaceholder("documents"),
    ])

    context_evaluator = prompt | llm | StrOutputParser()

    document_messages = [HumanMessage(content=doc.page_content) for doc in documents]

    resp = context_evaluator.invoke({"question": question, "documents": document_messages}).strip().upper()
    if resp == "YES":
        return {"documents": documents}
    if resp == "NO":
        return {"documents": []}
    raise ValueError(resp)


def _generate_response_node(state: RAGGraphState, llm: BaseChatModel) -> RAGGraphState:
    """Generates a response using the LLM based on retrieved
    documents."""
    question = state["question"]
    documents = state["documents"]
    # Prompt template from the PDF
    template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    # Format the context from the documents
    context = "\n\n".join([doc.page_content for doc in documents])
    # Create the RAG chain
    rag_chain = prompt | llm | StrOutputParser()
    # Invoke the chain
    generation = rag_chain.invoke({"context": context, "question": question})
    return {"generation": generation}


def _get_llm() -> BaseChatModel:
    return create_openai_llm(get_keys())

def _create_graph() -> CompiledStateGraph:
    llm = _get_llm()
    retriever = _get_retriever()

    workflow = StateGraph(RAGGraphState)
    # Add nodes
    workflow.add_node("retrieve", partial(_retrieve_documents_node, retriever=retriever, llm=llm))
    workflow.add_node("generate", partial(_generate_response_node, llm=llm))
    # Set the entry point
    workflow.set_entry_point("retrieve")
    # Add edges (transitions)
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    # Compile the graph
    app = workflow.compile()

    return app


def _main():
    graph = _create_graph()
    questions = (
        "What did the president say about Justice Breyer?",
        "What did the president say about the economy?",
        "Who is Martin Luther King?")

    for query in questions:
        print(f"******************** {query} ********************")
        inputs = {"question": query}
        for s in graph.stream(inputs):
            print(s)


if __name__ == '__main__':
    _main()
