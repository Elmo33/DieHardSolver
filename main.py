from langchain_community.llms import LlamaCpp
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from collections import deque
from typing import Tuple, List

# Corrected model path formatting
model_path = "/root/DieHardSolver/7B/qwen1_5-7b-chat-q2_k.gguf"

llm = LlamaCpp(
    model_path=model_path,
    n_ctx=1024,
    n_threads=16,
    n_batch=256,
    use_mlock=True,
    use_mmap=True
)

# Die Hard Solver Function
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
            (jug1, j2), (j1, jug2),  # Fill jugs
            (0, j2), (j1, 0),        # Empty jugs
            (j1 - min(j1, jug2 - j2), j2 + min(j1, jug2 - j2)),  # Pour jug1 -> jug2
            (j1 + min(j2, jug1 - j1), j2 - min(j2, jug1 - j1)),  # Pour jug2 -> jug1
        ]

        for new_j1, new_j2 in possible_moves:
            if (new_j1, new_j2) not in visited:
                queue.append((new_j1, new_j2, steps + [(j1, j2)]))

    return "No solution found"

# Improved execution function
def execute_die_hard_tool(input_str: str):
    print(f"DEBUG: Received input -> {input_str}")
    try:
        input_data = eval(input_str)  # Convert string to dictionary safely
        if not isinstance(input_data, dict) or "jugs" not in input_data or "target" not in input_data:
            raise ValueError("Invalid input format: expected {'jugs': [int, int], 'target': int}")

        jug1, jug2 = map(int, input_data["jugs"])
        target = int(input_data["target"])

        result = die_hard_solver(jug1, jug2, target)
        return f"DEBUG: Solution steps -> {result}"
    except Exception as e:
        return f"DEBUG: Error parsing input -> {str(e)}"

# Define LangChain Tool
die_hard_tool = Tool(
    name="DieHardSolver",
    func=execute_die_hard_tool,
    description="Solves the Die Hard problem given two jug sizes and a target amount of water."
)

# Updated LangChain agent with forced tool usage
agent = initialize_agent(
    tools=[die_hard_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history"),
    handle_parsing_errors=True
)

# Ensure the agent is always directed to use tools
agent.prompt = """You are an AI assistant that solves the Die Hard water jug problem. You must always use the DieHardSolver tool for calculations.

**Instructions:**
1. If the user provides jug sizes and a target amount, call `DieHardSolver` with the correct parameters.
2. If the input is incorrect, provide guidance but **do not** attempt to solve it yourself—always call the tool.

**Example Input:**
User: "Solve the Die Hard problem with jugs 3 and 5 to get 4 liters."
Agent Thought: "The user has provided a valid problem statement."
Action: DieHardSolver
Action Input: {"jugs": [3, 5], "target": 4}

**Example Incorrect Input Handling:**
User: "Solve the problem with jugs three and five."
Agent Thought: "The user provided an unclear request."
Response: "Please provide numeric jug sizes and target amount in liters."

Now proceed with the request:
"""

# Execute agent action
def execute_action(input_str: str):
    print("DEBUG: Executing agent...")
    response = agent.invoke({"input": input_str})
    print(f"DEBUG: Agent response -> {response}")

# Example usage:
execute_action("Solve the Die Hard problem with jugs 3 and 5 to get 4 liters.")
