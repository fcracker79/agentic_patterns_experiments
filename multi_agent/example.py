import operator
from pprint import pprint
from typing import Annotated, Literal

import nest_asyncio
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool as langchain_tool
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from typing_extensions import TypedDict

from common.keys import get_keys
from common.llm import create_openai_llm

_MAX_ITERATIONS = 6

# ── Tools ─────────────────────────────────────────────────────────────────────

@langchain_tool
def search_web(query: str) -> str:
    """Search the web for factual information about a topic."""
    print(f"\n  [tool:search_web] query='{query}'")
    db = {
        "python": "Python is a high-level, general-purpose language created by Guido van Rossum in 1991.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor LLM applications, built on LangChain.",
        "multi-agent": "Multi-agent systems use multiple AI agents that cooperate to solve complex tasks by dividing work.",
        "langchain": "LangChain is a framework for building applications powered by large language models.",
    }
    for key, value in db.items():
        if key in query.lower():
            return value
    return f"No specific result for '{query}', but it is an interesting topic."


@langchain_tool
def format_report(title: str, content: str) -> str:
    """Format raw content into a structured markdown report."""
    print(f"\n  [tool:format_report] title='{title}'")
    return f"# {title}\n\n{content}\n\n---\n*Report by the Writer Agent*"


# ── Supervisor structured output ──────────────────────────────────────────────

class SupervisorDecision(BaseModel):
    next: Literal["researcher", "writer", "FINISH"]
    reasoning: str


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # operator.add defines the behavior of merging state with returned node result.
    # messages = messages + returned_messages_by_node
    messages: Annotated[list[BaseMessage], operator.add]
    next: str
    iterations: int
    success: bool


# ── Nodes ─────────────────────────────────────────────────────────────────────

def make_supervisor_node(llm):
    system = SystemMessage(content=(
        "You are a supervisor managing two agents: 'researcher' and 'writer'.\n"
        "- Use 'researcher' to gather factual information.\n"
        "- Use 'writer' to format gathered information into a final report.\n"
        "- Use 'FINISH' when the task is fully complete.\n"
        "Typical flow: researcher → writer → FINISH."
    ))
    structured_llm = llm.with_structured_output(SupervisorDecision)

    def supervisor(state: AgentState) -> dict:
        iterations = state.get("iterations", 0) + 1
        if iterations >= _MAX_ITERATIONS:
            return {"next": "failed", "iterations": iterations, 'success': False}

        decision = structured_llm.invoke([system] + state["messages"])
        print(f"\n[supervisor] → {decision.next}  reason: {decision.reasoning}")
        return {"next": decision.next, "iterations": iterations, 'success': True}

    return supervisor


def make_agent_node(llm, tools: list, name: str):
    agent = create_agent(llm, tools)

    def node(state: AgentState) -> dict:
        print(f"\n[{name}] running...")
        result = agent.invoke(state)
        last_message = result["messages"][-1]
        last_message.name = name
        # The rest of the context is kept: `messages` have the annotation `operator.add`,
        # so they will be concatenated across iterations.
        return {"messages": [last_message]}

    return node


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph(llm):
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", make_supervisor_node(llm))
    graph.add_node("researcher", make_agent_node(llm, [search_web], "researcher"))
    graph.add_node("writer", make_agent_node(llm, [format_report], "writer"))

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {"researcher": "researcher", "writer": "writer", "FINISH": END, "failed": END},
    )
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")

    return graph.compile()


# ── Main ──────────────────────────────────────────────────────────────────────

def _main():
    keys = get_keys()
    llm = create_openai_llm(keys)
    app = build_graph(llm)

    task = "Research what LangGraph is and write a short report about it."
    print(f"Task: {task}\n" + "=" * 60)

    result = app.invoke({"messages": [HumanMessage(content=task)]})

    pprint(result)
    if result['success']:
        print("\n" + "=" * 60 + " FINAL OUTPUT " + "=" * 60)
        # The response is provided by researcher or writer: they both have the whole context
        # and a specific tool, if needed. The supervisor instead is limited to what it can
        # return.
        for msg in result["messages"]:
            name = getattr(msg, "name", None) or type(msg).__name__
            print(f"\n[{name}]\n{msg.content}")
    else:
        print("\n" + "=" * 60 + " FAILED " + "=" * 60)


if __name__ == "__main__":
    nest_asyncio.apply()
    _main()
