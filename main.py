from langchain_community.llms import LlamaCpp
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from collections import deque
from typing import Tuple, List
model_path = "/root/DieHardSolver/7B/qwen1_5-7b-chat-q2_k.gguf"
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=1024,
    n_threads=16,
    n_batch=256,
    use_mlock=True,
    use_mmap=True
)

def die_hard_solver(jug1: int, jug2: int, target: int):
    visited = set()
    queue = deque([(0, 0, [])])

    while queue:
        j1, j2, steps = queue.popleft()
        if (j1, j2) in visited:
            continue

        visited.add((j1, j2))

        if j1 == target or j2 == target:
            return steps + [(j1, j2)]

        possible_moves = [
            (jug1, j2), (j1, jug2),  # fill jug1 or jug2
            (0, j2), (j1, 0),        # empty jug1 or jug2
            (j1 - min(j1, jug2 - j2), j2 + min(j1, jug2 - j2)),  # jug1 -> jug2
            (j1 + min(j2, jug1 - j1), j2 - min(j2, jug1 - j1)),  # jug2 -> jug1
        ]

        for new_j1, new_j2 in possible_moves:
            if (new_j1, new_j2) not in visited:
                queue.append((new_j1, new_j2, steps + [(j1, j2)]))

    return "No solution found"

# Debug-friendly execution
def execute_die_hard_tool(input_str: str):
    print(f"DEBUG: Received input -> {input_str}")
    try:
        jug1, jug2, target = map(int, input_str.split())
        result = die_hard_solver(jug1, jug2, target)
        return f"DEBUG: Solution steps -> {result}"
    except ValueError as e:
        return f"DEBUG: Error parsing input -> {str(e)}"

# Define LangChain Tool
die_hard_tool = Tool(
    name="DieHardSolver",
    func=execute_die_hard_tool,
    description="Solve Die Hard problem given jug sizes and target"
)

# Initialize agent with debugging output
agent = initialize_agent(
    tools=[die_hard_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history"),
    handle_parsing_errors=True
)

# Execute agent action
def execute_action(input_str: str):
    print("DEBUG: Executing agent...")
    response = agent.invoke({"input": input_str})
    print(f"DEBUG: Agent response -> {response}")

# Example usage:
execute_action("Solve the Die Hard problem with jugs 3 and 5 to get 4 liters.")