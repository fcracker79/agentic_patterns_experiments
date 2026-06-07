import asyncio

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

from common.llm import create_ollama_llm


async def _main():
    llm = create_ollama_llm('llama3.2:3b')

    # --- Define Independent Chains ---
    # These three chains represent distinct tasks that can be executed in parallel.
    summarize_chain: Runnable = (
            ChatPromptTemplate.from_messages([
                ("system", "Summarize the following topic concisely:"),
                ("user", "{topic}")
            ])
            | llm
            | StrOutputParser()
    )
    questions_chain: Runnable = (
            ChatPromptTemplate.from_messages([
                ("system", "Generate three interesting questions about the following topic:"),
                ("user", "{topic}")
            ])
            | llm
            | StrOutputParser()
    )
    terms_chain: Runnable = (
            ChatPromptTemplate.from_messages([
                ("system", "Identify 5-10 key terms from the following topic, separated by commas:"),
                ("user", "{topic}")
            ])
            | llm
            | StrOutputParser()
    )

    # --- Build the Parallel + Synthesis Chain ---
    # 1. Define the block of tasks to run in parallel. The results of these,
    # along with the original topic, will be fed into the next step.
    map_chain = RunnableParallel(
        {
            "summary": summarize_chain,
            "questions": questions_chain,
            "key_terms": terms_chain,
            "topic": RunnablePassthrough(),  # Pass the original topic through
        }
    )

    # 2. Define the final synthesis prompt which will combine the parallel results.
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", """Based on the following information:
    Summary: {summary}
    Related Questions: {questions}
    Key Terms: {key_terms}
    Synthesize a comprehensive answer."""),
        ("user", "Original topic: {topic}")
    ])
    synthesis_chain = synthesis_prompt | llm | StrOutputParser()

    # 3. Construct the full chain by piping the parallel results directly
    # into the synthesis prompt, followed by the LLM and output parser.
    full_parallel_chain = map_chain | synthesis_chain

    topic = "The history of space exploration"
    response = await full_parallel_chain.ainvoke(topic)
    print("\n--- Final Response ---")
    print(response)


if __name__ == "__main__":
    asyncio.run(_main())
