import time

from langchain_community.llms.llamacpp import LlamaCpp
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool


class DieHardProblem:
    def __init__(self):
        self.small = 0  # 3-gallon jug
        self.big = 0  # 5-gallon jug

    def fill_small(self):
        self.small = 3

    def fill_big(self):
        self.big = 5

    def empty_small(self):
        self.small = 0

    def empty_big(self):
        self.big = 0

    def pour_small_into_big(self):
        old_big = self.big
        self.big = min(5, self.big + self.small)
        self.small = self.small - (self.big - old_big)

    def pour_big_into_small(self):
        old_small = self.small
        self.small = min(3, self.small + self.big)
        self.big = self.big - (self.small - old_small)

    def physics_of_jugs(self):
        assert 0 <= self.small <= 3, "Small jug out of bounds"
        assert 0 <= self.big <= 5, "Big jug out of bounds"

    def die_hard_problem_not_solved(self):
        assert self.big != 4, f"Puzzle solved: big jug contains {self.big} gallons"

    def reset(self):
        self.small = 0
        self.big = 0

    def state(self):
        return (self.small, self.big)


problem = DieHardProblem()


@tool
def reset_to_initial_state():
    """
    Resets both jugs (small and big) to 0 gallons.
    """
    problem.reset()
    return f"Jugs reset to initial state: {problem.state()}"


@tool
def fill_small_jug():
    """
    Fills the small jug (3 gallons) to its full capacity.
    """
    problem.fill_small()
    return f"Filled small jug. Current state: {problem.state()}"


@tool
def fill_big_jug():
    """
    Fills the big jug (5 gallons) to its full capacity.
    """
    problem.fill_big()
    return f"Filled big jug. Current state: {problem.state()}"


@tool
def empty_small_jug():
    """
    Empties the small jug (3 gallons).
    """
    problem.empty_small()
    return f"Emptied small jug. Current state: {problem.state()}"


@tool
def empty_big_jug():
    """
    Empties the big jug (5 gallons).
    """
    problem.empty_big()
    return f"Emptied big jug. Current state: {problem.state()}"


@tool
def pour_small_into_big_jug():
    """
    Pours water from the small jug into the big jug until the big jug is full or the small jug is empty.
    """
    problem.pour_small_into_big()
    return f"Poured small into big. Current state: {problem.state()}"


@tool
def pour_big_into_small_jug():
    """
    Pours water from the big jug into the small jug until the small jug is full or the big jug is empty.
    """
    problem.pour_big_into_small()
    return f"Poured big into small. Current state: {problem.state()}"


@tool
def get_state():
    """
    Returns the current state of both jugs (small and big).
    """
    return f"Current state: {problem.state()}"


@tool
def reset_problem():
    """
    Resets the DieHardProblem instance and returns the current state.
    """
    global problem
    problem = DieHardProblem()
    return f"Problem reset. Current state: {problem.state()}"


def main():
    model_path = "/root/DieHardSolver/7B/qwen1_5-7b-chat-q2_k.gguf"
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=1024,
        n_threads=16,
        n_batch=256,
        use_mlock=True,
        use_mmap=True
    )

    tools = [
        fill_small_jug,
        fill_big_jug,
        empty_small_jug,
        empty_big_jug,
        pour_small_into_big_jug,
        pour_big_into_small_jug,
        get_state,
        reset_problem,
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    prompt = """You are given a puzzle called the Die Hard water jug problem. You have two jugs: a 3-gallon jug (small) and a 5-gallon jug (big). The goal is to obtain exactly 4 gallons in the 5-gallon jug using the following tools:

    - fill_small_jug – Fills the small jug to its 3-gallon capacity.
    - fill_big_jug – Fills the big jug to its 5-gallon capacity.
    - empty_small_jug – Empties the small jug.
    - empty_big_jug – Empties the big jug.
    - pour_small_into_big_jug – Pours water from the small jug into the big jug until the big jug is full or the small jug is empty.
    - pour_big_into_small_jug – Pours water from the big jug into the small jug until the small jug is full or the big jug is empty.
    - get_state – Returns the current state of both jugs.
    - reset_problem – Resets both jugs to 0 gallons.
    
    **Instructions:**
    - Your goal is to fill the 5-gallon jug with exactly 4 gallons of water.
    - You may use the tools in any order, but **your first step should always be to reset both jugs to 0 gallons** by using the reset_problem tool. You can only use this tool once at the beginning.
    - Provide the sequence of operations needed to achieve the goal. Each operation must be clearly defined and must lead toward reaching exactly 4 gallons in the 5-gallon jug.
    
    Make sure to plan your steps carefully, and use the tools strategically to reach the goal in the fewest moves possible."
    """


    max_steps = 30  # Limit the maximum number of iterations
    steps_taken = 0

    while steps_taken < max_steps:
        response = agent.invoke(prompt)
        print(response["text"])

        if problem.state()[1] == 4:
            for index, action in enumerate(response["intermediate_steps"]):
                index += 1
                print(f"{index}.{action[0].tool}")
            break

        steps_taken += 1
        if steps_taken == max_steps:
            print("Maximum steps reached, could not solve the problem.")
            break


if __name__ == "__main__":
    main()
