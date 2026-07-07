import asyncio
import json

import nest_asyncio
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient

from common.keys import get_keys
from common.llm import create_openai_llm

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"


async def _main():
    keys = get_keys()
    llm = create_openai_llm(keys)

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
        # LLMs have a bias towards tools. It requires a strong system prompt to avoid misuse or wrong behavior.
        ("system", "You are a helpful assistant. Use tools only to fetch or act on external "
                   "data that you don't already have. When no tool fits part of a task, "
                   "complete that part yourself using your own knowledge instead of forcing "
                   "a tool call. Once every part of the user's request has been addressed, "
                   "give exactly one final answer that recaps the results for all of them "
                   "together — never submit a final answer that covers only part of the request."),
        # Resources must be passed explicitly
        ("assistant", f"Your enemies are {enemies}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # This will execute tool calls
    agent = create_tool_calling_agent(llm, tools, prompt)
    # This is our interface with the LLM
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = await agent_executor.ainvoke({"input": "For each of my friends, create a personalized greeting. For each of my enemies, create a personalized offense"})
    print("\n--- Final Response ---")
    print(response["output"])


if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(_main())
