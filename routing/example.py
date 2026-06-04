from pprint import pprint

from common.keys import get_keys
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

from common.llm import create_openai_llm

def _extract_request_message(request: dict) -> str:
    return request['request']['user_request']


# --- Define Simulated Sub-Agent Handlers (equivalent to ADKsub_agents) ---
def booking_handler(request: dict) -> str:
    """Simulates the Booking Agent handling a request."""
    request_msg = _extract_request_message(request)
    print("\n--- DELEGATING TO BOOKING HANDLER ---")
    return f"Booking Handler processed request: '{request_msg}'. Result: Simulated booking action."


def info_handler(request: dict) -> str:
    """Simulates the Info Agent handling a request."""
    request_msg = _extract_request_message(request)
    print("\n--- DELEGATING TO INFO HANDLER ---")
    return f"Info Handler processed request: '{request_msg}'. Result: Simulated information retrieval."


def unclear_handler(request: dict) -> str:
    """Handles requests that couldn't be delegated."""
    request_msg = _extract_request_message(request)
    print("\n--- HANDLING UNCLEAR REQUEST ---")
    return f"Coordinator could not delegate request: '{request_msg}'. Please clarify."


# --- Define Coordinator Router Chain (equivalent to ADK coordinator's instruction) ---
# This chain decides which handler to delegate to.
coordinator_router_prompt = ChatPromptTemplate.from_messages([
    ("system", """Analyze the user's request and determine which
specialist handler should process it.
- If the request is related to booking flights or hotels,
output 'booker'.
- For all other general information questions, output 'info'.
- If the request is unclear or doesn't fit either category,
output 'unclear'.
ONLY output one word: 'booker', 'info', or 'unclear'."""),
    ("user", "{user_request}")
])


def _main():
    keys = get_keys()
    llm = create_openai_llm(keys)
    coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()

    # --- Define the Delegation Logic (equivalent to ADK's Auto-Flow based on sub_agents) ---
    # Use RunnableBranch to route based on the router chain's output.
    # Define the branches for the RunnableBranch
    branches = {
        "booker": RunnablePassthrough.assign(output=booking_handler),
        "info": RunnablePassthrough.assign(output=info_handler),
        "unclear": RunnablePassthrough.assign(output=unclear_handler),
    }

    # Create the RunnableBranch. It takes the output of the router chain
    # and routes the original input ('request') to the corresponding handler.
    delegation_branch = RunnableBranch(
        (_debug_branch, branches["booker"]),
        (lambda x: x['decision'].strip() == 'booker', branches["booker"]),
        # Added .strip()
        (lambda x: x['decision'].strip() == 'info', branches["info"]),
        # Added .strip()
        branches["unclear"]  # Default branch for 'unclear' or any other output
    )

    # Combine the router chain and the delegation branch into a single runnable
    # The router chain's output ('decision') is passed along with the original input('request')
    # to the delegation_branch.
    coordinator_agent = ({
                            "decision": coordinator_router_chain,
                            "request": RunnablePassthrough()
                        } | delegation_branch
                        # Here I could use also RunnableLambda or even operator.itemgetter('output')
                         | (lambda x: x['output']))  # Extract the final output
    print("--- Running with a booking request ---")
    request_a = "Book me a flight to London."
    result_a = coordinator_agent.invoke({"user_request": request_a})
    print(f"Final Result A: {result_a}")

def _debug_branch(x):
    print("This is the input of the RunnableBranch")
    pprint(x)
    return False


if __name__ == '__main__':
    _main()
