from langchain_community.llms import LlamaCpp
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from collections import deque
from typing import Dict

# Define model path
model_path = "/root/DieHardSolver/7B/qwen1_5-7b-chat-q2_k.gguf"

# Load the Llama model
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=1024,
    n_threads=16,
    n_batch=512,
    use_mlock=True,
    use_mmap=True,
    verbose=True,
    streaming=True  # Enable streaming for better responses
)

# Define the Die Hard water jug problem solver function
def die_hard_solver(jug1: int, jug2: int, target: int):
    """Solves the Die Hard problem given two jug capacities and a target amount."""
    visited = set()
    queue = deque([(0, 0, [])])  # (jug1_state, jug2_state, steps)

    while queue:
        j1, j2, steps = queue.popleft()
        if (j1, j2) in visited:
            continue
        visited.add((j1, j2))

        if j1 == target or j2 == target:
            return steps + [(j1, j2)]

        actions = [
            (jug1, j2),  # Fill jug1
            (j1, jug2),  # Fill jug2
            (0, j2),  # Empty jug1
            (j1, 0),  # Empty jug2
            (j1 - min(j1, jug2 - j2), j2 + min(j1, jug2 - j2)),  # Pour jug1 → jug2
            (j1 + min(j2, jug1 - j1), j2 - min(j2, jug1 - j1))  # Pour jug2 → jug1
        ]

        for new_j1, new_j2 in actions:
            queue.append((new_j1, new_j2, steps + [(j1, j2)]))

    return "No solution found"

# Properly structured tool function
def solve_die_hard(jug1: int, jug2: int, target: int) -> str:
    """Solves the Die Hard problem with structured inputs."""
    print(f"DEBUG: Received input -> jug1: {jug1}, jug2: {jug2}, target: {target}")  # Debugging
    result = die_hard_solver(jug1, jug2, target)
    return f"Solution steps: {result}" if isinstance(result, list) else result

# Define a properly structured tool
die_hard_tool = StructuredTool.from_function(
    solve_die_hard,
    name="DieHardSolver",
    description="Solves the Die Hard problem given two jug sizes and a target amount.",
)

# Initialize agent with memory
memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools=[die_hard_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,  # Avoid crashing due to format issues
)

# Run the agent using structured input
response = agent.invoke(
    {
        "input": "Solve the Die Hard problem with jugs 3 and 5 to get 4 liters."
    }
)
print(response)
