import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from common.keys import get_keys
from common.llm import create_openai_llm


def _main():
    keys = get_keys()
    llm = create_openai_llm(keys)

    # --- Prompt 1: Extract Information ---
    prompt_extract = ChatPromptTemplate.from_template(
        "Extract the technical specifications from the following text:\n\n{text_input}"
    )

    # --- Prompt 2: Transform to JSON ---
    prompt_transform = ChatPromptTemplate.from_template(
        "Transform the following specifications into a JSON object with 'cpu', 'memory', and 'storage' as keys:\n\n {specifications}"
    )

    prompt_verification = ChatPromptTemplate.from_template(
        "Verify that 'cpu', 'memory', and 'storage' in {transformations} are consistent with {specifications}. Response must be 'true' or 'false'"
    )

    # --- Build the Chain using LCEL ---
    # The StrOutputParser() converts the LLM's message output to a simple string.
    extraction_chain = prompt_extract | llm | StrOutputParser()
    transformation_chain = prompt_transform | llm| StrOutputParser()
    verification_chain = prompt_verification | llm | StrOutputParser()

    # The full chain passes the output of the extraction chain into the 'specifications' variable for the transformation prompt.
    # Each step adds a key to the running dict (instead of replacing it),
    # so the verification prompt can still see 'specifications'.
    full_chain = (
            {"specifications": extraction_chain}
            | RunnablePassthrough.assign(transformations=transformation_chain)
            | RunnablePassthrough.assign(verification=verification_chain)
            | RunnableLambda(lambda x: {"result": json.loads(x["transformations"]), "valid": x["verification"] == "true"})
    )

    # --- Run the Chain ---
    input_text = "The new laptop model features a 3.5 GHz octa-core processor, 16 GB of RAM, and a 1 TB NVMe SSD."
    # Execute the chain with the input text dictionary.
    final_result = full_chain.invoke({"text_input": input_text})
    print("\n--- Final JSON Output ---")
    print(final_result)


if __name__ == '__main__':
    _main()
