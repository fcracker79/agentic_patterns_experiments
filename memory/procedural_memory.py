from langchain_core.messages import AnyMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from common.keys import get_keys
from common.llm import create_openai_llm

NAMESPACE = ("agent",)
INSTRUCTIONS_KEY = "instructions"
DEFAULT_INSTRUCTIONS = "You are a helpful assistant. Be clear and accurate."

_answer_prompt = ChatPromptTemplate.from_template(
    "{instructions}\n\nConversation:\n{conversation}"
)

_update_prompt = ChatPromptTemplate.from_template(
    """You are improving an AI assistant's system instructions based on a conversation.

Current instructions:
{instructions}

Conversation:
{conversation}

Rewrite the instructions to make the assistant more helpful based on what you observed.
Output ONLY the new instructions, no commentary."""
)


def _fmt_messages(messages: list[AnyMessage]) -> str:
    return "\n".join(f"{m.type.upper()}: {m.content}" for m in messages)


def call_model(state: MessagesState, store: BaseStore):
    item = store.get(NAMESPACE, INSTRUCTIONS_KEY)
    instructions = item.value["instructions"] if item else DEFAULT_INSTRUCTIONS

    llm = create_openai_llm(get_keys())
    response = (_answer_prompt | llm | StrOutputParser()).invoke({
        "instructions": instructions,
        "conversation": _fmt_messages(state["messages"]),
    })
    return {"messages": [{"role": "assistant", "content": response}]}


def update_instructions(state: MessagesState, store: BaseStore):
    item = store.get(NAMESPACE, INSTRUCTIONS_KEY)
    current = item.value["instructions"] if item else DEFAULT_INSTRUCTIONS

    llm = create_openai_llm(get_keys())
    new_instructions = (_update_prompt | llm | StrOutputParser()).invoke({
        "instructions": current,
        "conversation": _fmt_messages(state["messages"]),
    })
    store.put(NAMESPACE, INSTRUCTIONS_KEY, {"instructions": new_instructions})


def _main():
    store = InMemoryStore()  # no embeddings needed: accesso diretto per chiave

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(update_instructions)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", "update_instructions")
    builder.add_edge("update_instructions", END)
    graph = builder.compile(store=store)

    turns = [
        "Hi! Can you explain what a Python decorator is?",
        "I prefer very short answers with a minimal code example, no prose.",
        "What is a context manager?",
        "What is a generator?",
    ]

    for msg in turns:
        print(f"\nUser: {msg}")
        result = graph.invoke({"messages": [{"role": "user", "content": msg}]})
        print(f"Assistant: {result['messages'][-1].content}")
        item = store.get(NAMESPACE, INSTRUCTIONS_KEY)
        if item:
            print(f"[Instructions: {item.value['instructions'][:120].strip()}...]")


if __name__ == "__main__":
    _main()
