from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from common.keys import get_keys
from common.llm import create_openai_llm


def chat_verify_with_cfg(app: Runnable, cfg: dict):
    def chat(question: str) -> str:
        result = app.invoke({"messages": [HumanMessage(content=question)]}, config=cfg)
        return result["messages"][-1].content

    print(chat(f"Tell me who I am and recap what we talked about"))


def chat_with_cfg(app: Runnable, therad_id: str, cfg: dict):
    def chat(question: str) -> str:
        result = app.invoke({"messages": [HumanMessage(content=question)]}, config=cfg)
        return result["messages"][-1].content

    print(chat(f"I want to book a flight for {therad_id}."))
    print(chat(f"My name is Sam-{therad_id}, by the way."))
    print(chat("What was my name again?"))


def print_memory(memory_saver: BaseCheckpointSaver, cfg: dict):
    for m in memory_saver.get(cfg)["channel_values"]["messages"]:
        print(f'\t{m.content}')

def _main():
    llm = create_openai_llm(get_keys())

    system_message = SystemMessage(content="You are a helpful travel agent.")

    def call_model(state: MessagesState):
        response = llm.invoke([system_message] + state["messages"])
        return {"messages": response}

    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory_saver = MemorySaver()
    app = workflow.compile(checkpointer=memory_saver)

    cfg1 = {"configurable": {"thread_id": 'travel'}}
    cfg2 = {"configurable": {"thread_id": 'business'}}
    chat_with_cfg(app, 'travel', cfg1)
    chat_with_cfg(app, 'business', cfg2)

    print('CFG1')
    chat_verify_with_cfg(app, cfg1)
    print_memory(memory_saver, cfg1)

    print('CFG2')
    chat_verify_with_cfg(app, cfg2)
    print_memory(memory_saver, cfg2)


if __name__ == "__main__":
    _main()
