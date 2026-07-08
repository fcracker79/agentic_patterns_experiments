import asyncio
import json

import nest_asyncio
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient

from common.keys import get_keys
from common.llm import create_openai_llm

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"
MAX_REFLECTION_ITERATIONS = 3

_REFLECTOR_PROMPT = "You are a meticulous editor reviewing an assistant's response against the " \
                     "original request. Check that every friend received a personalized greeting " \
                     "and every enemy received a personalized farewell." \
                     "If the response fully satisfies the request, respond with the single " \
                     "phrase 'RESPONSE_IS_PERFECT'. Otherwise, respond with a concise bulleted " \
                     "list of what is missing or needs improvement."

# LLMs have a bias towards tools. It requires a strong system prompt to avoid misuse or wrong behavior.
_STRICT_PROMPT = "You are a helpful assistant. Use tools only to fetch or act on external "\
                   "data that you don't already have. When no tool fits part of a task, "\
                   "complete that part yourself using your own knowledge instead of forcing "\
                   "a tool call. Once every part of the user's request has been addressed, "\
                   "give exactly one final answer that recaps the results for all of them "\
                   "together — never submit a final answer that covers only part of the request."


_LESS_STRICT_PROMPT = "You are a helpful assistant. Use tools only to fetch or act on external "\
                   "data that you don't already have. When no tool fits part of a task, "\
                   "complete that part yourself using your own knowledge instead of forcing "\
                   "a tool call."

_SYSTEM_PROMPT = _STRICT_PROMPT


async def _reflect(llm, original_request: str, draft: str) -> str:
    critique = await llm.ainvoke([
        SystemMessage(content=_REFLECTOR_PROMPT),
        HumanMessage(content=f"Original request:\n{original_request}\n\nResponse to review:\n{draft}"),
    ])
    return critique.content


async def _main():
    keys = get_keys()
    llm = create_openai_llm(keys, model='gpt-5.2')

    client = MultiServerMCPClient(
        {
            "demo_server": {
                "url": MCP_SERVER_URL,
                "transport": "http",
            }
        }
    )
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tool(s) from MCP server: {[t.name for t in tools]}")

    resources = await client.get_resources()
    print(f"Loaded {len(resources)} resource(s) from MCP server: {resources}")
    enemies_resource = resources[0]
    enemies = [e["name"] for e in json.loads(enemies_resource.data)]

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        # Resources must be passed explicitly
        ("assistant", f"Your enemies are {enemies}"),
        # "placeholder" splices in real messages with their own roles; a System/AIMessagePromptTemplate
        # would just stringify the list into one flattened message and lose turn structure.
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # This will execute tool calls
    agent = create_tool_calling_agent(llm, tools, prompt)
    # This is our interface with the LLM
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    user_request = ("For each of my friends, create a personalized greeting. "
                    "For each of my enemies, create a personalized farewell.")
    # chat_history accumulates the real conversation turns (request, draft, refine request, draft, ...)
    # so each refine iteration sees the full context instead of a single re-stated string.
    chat_history = []
    current_input = user_request
    draft = None

    worked = False
    for i in range(MAX_REFLECTION_ITERATIONS):
        response = await agent_executor.ainvoke({"input": current_input, "chat_history": chat_history})
        draft = response["output"]
        print(f"\n--- Draft {i + 1} ---\n{draft}")

        chat_history.append(HumanMessage(content=current_input))
        chat_history.append(AIMessage(content=draft))

        critique = await _reflect(llm, user_request, draft)
        if "RESPONSE_IS_PERFECT" in critique:
            print("\n--- Reflection ---\nNo further critiques. The response is satisfactory.")
            worked = True
            break

        print(f"\n--- Reflection critique ---\n{critique}")
        current_input = f"Please refine your previous response based on this critique:\n{critique}"

    print("\n--- Final Response ---")
    if worked:
        print(draft)
    else:
        print(f"NOT WORKED! Parial response: {draft}")

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(_main())
