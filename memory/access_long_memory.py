import asyncio
from dataclasses import dataclass
from pprint import pprint
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, AIMessage, AIMessageChunk, ChatMessage, ChatMessageChunk, \
    FunctionMessage, FunctionMessageChunk, HumanMessage, HumanMessageChunk, SystemMessage, SystemMessageChunk, \
    ToolMessage, ToolMessageChunk
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, MessagesState, START
import uuid

from langgraph.store.memory import InMemoryStore

from common.keys import get_keys
from common.llm import create_openai_llm


@dataclass
class Context:
    user_id: str

async def call_model(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")

    last_message: AnyMessage = state['messages'][-1]
    llm = create_openai_llm(get_keys())

    # Search for relevant memories
    memories = await runtime.store.asearch(
        namespace, query=last_message.content, limit=3
    )
    info = "\n".join([d.value["data"] for d in memories])

    answer_chain = ChatPromptTemplate.from_template(
        '''
        relevant memories: {info}
        question: {question}

        Given the relevant memories, answer the given question.
        '''
    ) | llm | StrOutputParser()

    response = answer_chain.invoke({'info': info, 'question': last_message.content})

    await _update_long_term_memory(last_message, llm, namespace, response, runtime)


async def _update_long_term_memory(
        last_message: AnyMessage,
        llm: BaseChatModel, namespace: tuple[str, Literal["memories"]],
        response: str, runtime: Runtime[Context]):
    # Extract key facts from this exchange before storing
    extraction_chain = ChatPromptTemplate.from_template(
        '''
        A user asked the following question:
        {question}

        The assistant responded:
        {response}

        Extract a concise list of key facts, preferences, or insights about the user or topic
        that are worth remembering for future conversations. Write one fact per line.
        Do not include filler — only information that would be useful to recall later.
        '''
    ) | llm | StrOutputParser()

    facts = extraction_chain.invoke({'question': last_message.content, 'response': response})

    await runtime.store.aput(
        namespace, str(uuid.uuid4()), {"data": facts}
    )


async def _main():
    builder = StateGraph(MessagesState, context_schema=Context)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")

    store = InMemoryStore()
    graph = builder.compile(store=store)

    # Pass context at invocation time
    await graph.ainvoke(
        {"messages": [{"role": "user", "content": "who are you?"}]},
        {"configurable": {"thread_id": "thread1"}}, # not relevant, we did not add checkpointer
        context=Context(user_id="user1"),
    )

    print("\n=== Memory Store Contents ===")
    namespace = ("user1", "memories")
    items = await store.asearch(namespace, query="hi", limit=100)
    for item in items:
        print(f"Key: {item.key}")
        pprint(item.value)


if __name__ == "__main__":
    asyncio.run(_main())
