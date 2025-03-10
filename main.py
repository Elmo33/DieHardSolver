from langchain.llms import Llama
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory

# Configure Qwen 7B optimized for CPX51 on Hetzner
llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen1.5-7B-Chat-GGUF",
    filename="qwen1_5-7b-chat-q2_k.gguf",
    n_ctx=1024,
    n_threads=16,  # Utilize all CPU cores on CPX51
    n_batch=512,
    use_mlock=True,
    use_mmap=True,
)


# Define the Die Hard water jug problem solver tool
def die_hard_solver(jug1: int, jug2: int, target: int):
    """Solves the Die Hard problem given two jug capacities and a target amount."""
    from collections import deque

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
            (jug1, j2), (j1, jug2), (0, j2), (j1, 0),
            (j1 - min(j1, jug2 - j2), j2 + min(j1, jug2 - j2)),
            (j1 + min(j2, jug1 - j1), j2 - min(j2, jug1 - j1))
        ]

        for new_j1, new_j2 in actions:
            queue.append((new_j1, new_j2, steps + [(j1, j2)]))

    return "No solution found"


# Create a tool for solving the Die Hard problem
die_hard_tool = Tool(
    name="DieHardSolver",
    func=lambda x: die_hard_solver(*map(int, x.split())),
    description="Solves the Die Hard problem given two jug sizes and a target amount. Input format: '3 5 4'",
)

# Initialize agent
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(
    tools=[die_hard_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)

# Example query
response = agent.run("Solve the Die Hard problem with jugs 3 and 5 to get 4 liters.")
print(response)