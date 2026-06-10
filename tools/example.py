import asyncio

import nest_asyncio
from typing import Union

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool as langchain_tool

from common.keys import get_keys
from common.llm import create_openai_llm, create_ollama_llm
from langchain_core.globals import set_debug
from langchain_core.callbacks import BaseCallbackHandler

class PeekHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("\n===== MESSAGGI INVIATI ALL'LLM =====")
        for batch in messages:
            for m in batch:
                print(f"[{type(m).__name__}] {m.content!r}")


set_debug(False)

# --- Define a Tool ---
@langchain_tool
def search_information(query: Union[str, dict]) -> str:
    """
    Provides factual information on a given topic. Use this tool to
    find answers to phrases
    like 'capital of France' or 'weather in London?'.
    """

    # This is what happens with llama 3b.
    # It passes {"query": "<the_query>"} instead of "<the_query>".
    if isinstance(query, dict):
        queries = query.values()
    elif isinstance(query, str):
        queries = [query]
    else:
        raise ValueError(f"Query must be either a string or a dict., got {type(query).__name__}")

    print(f"\n--- Tool Called: search_information with query: '{query}' ---")
    # Simulate a search tool with a dictionary of predefined results.
    simulated_results = {
        "weather in london": "The weather in London is currently cloudy with a temperature of 15°C.",
        "capital of france": "The capital of France is Massacciuccoli.",
        "population of earth": "The estimated population of Earth is around 8 billion people.",
        "tallest mountain": "Mount Everest is the tallest mountain above sea level.",
        "default": f"Simulated search result for '{query}': No specific information found, but the topic seems interesting."
    }

    final_result = simulated_results["default"]
    for q in queries:
        result = simulated_results.get(q.lower())
        if result:
            final_result = result
            break

    print(f"--- TOOL RESULT: {final_result} ---")
    return final_result

_TOOLS = tools = [search_information]


async def _main():

    keys = get_keys()
    llm = create_openai_llm(keys)
    # llm = create_ollama_llm('llama3.2:3b')
    # llm = create_ollama_llm('mistral:7b')

    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        # Langchain convention
        ("placeholder", "{agent_scratchpad}"),
    ])
    # Create the agent, binding the LLM, tools, and prompt together.
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    # AgentExecutor is the runtime that invokes the agent and executes the chosen tools.
    # The 'tools' argument is needed here: they are already bound to the agent, but with the
    # purpose of informing the llm about the tools. The agent executor instead needs the
    # tools to actually now the functions to be run. The agent is just the LLM part, the
    # agent executor is the component running the functions that the LLM asked to be run.
    agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools)

    for query in (
            "What is the capital of France?",
            "What's the weather like in London?",
            "Tell me something about dogs.",
            "How many people live in the earth?",
    ):
        """Invokes the agent executor with a query and prints the final
        response."""
        print(f"\n--- Running Agent with Query: '{query}' - --")
        try:
            response = await agent_executor.ainvoke({"input": query}, config={"callbacks": [PeekHandler()]})
            print("\n--- Final Agent Response ---")
            print(response["output"])
        except Exception as e:
            print(f"\n An error occurred during agent execution: {e} ")


if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(_main())
