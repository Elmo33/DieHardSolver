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
        # This assertion is meant to be violated when the puzzle is solved (i.e. when big == 4)
        assert self.big != 4, f"Puzzle solved: big jug contains {self.big} gallons"

    def reset(self):
        self.small = 0
        self.big = 0

    def state(self):
        return (self.small, self.big)


problem = DieHardProblem()


@tool
def reset_to_initial_state():
    """Reset the jugs to their initial state (both empty)."""
    problem.reset()
    return f"Jugs reset to initial state: {problem.state()}"


@tool
def fill_small_jug():
    """Fill the 3-gallon jug to capacity."""
    problem.fill_small()
    return f"Filled small jug. Current state: {problem.state()}"


@tool
def fill_big_jug():
    """Fill the 5-gallon jug to capacity."""
    problem.fill_big()
    return f"Filled big jug. Current state: {problem.state()}"


@tool
def empty_small_jug():
    """Empty the 3-gallon jug."""
    problem.empty_small()
    return f"Emptied small jug. Current state: {problem.state()}"


@tool
def empty_big_jug():
    """Empty the 5-gallon jug."""
    problem.empty_big()
    return f"Emptied big jug. Current state: {problem.state()}"


@tool
def pour_small_into_big_jug():
    """Pour water from the 3-gallon jug into the 5-gallon jug."""
    problem.pour_small_into_big()
    return f"Poured small into big. Current state: {problem.state()}"


@tool
def pour_big_into_small_jug():
    """Pour water from the 5-gallon jug into the 3-gallon jug."""
    problem.pour_big_into_small()
    return f"Poured big into small. Current state: {problem.state()}"


@tool
def get_state():
    """Return the current state of the jugs."""
    return f"Current state: {problem.state()}"


@tool
def reset_problem():
    """Reset the problem to its initial state (both jugs empty)."""
    global problem
    problem = DieHardProblem()
    return f"Problem reset. Current state: {problem.state()}"


def main():
    models = [
        # "deepseek-r1:8b",
        # "qwen2.5:7b",
        "qwen2.5-coder:14b",
        # "mistral"
    ]

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
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    prompt = (
        "You are given a Die Hard water jug problem with two jugs: one of capacity 3 gallons and one of 5 gallons. "
        "You have access to the following tools: fill_small_jug, fill_big_jug, empty_small_jug, empty_big_jug, pour_small_into_big_jug, "
        "pour_big_into_small_jug, get_state, and reset_problem. Your goal is to get exactly 4 gallons in the 5-gallon jug. "
        "Using the available tools, provide the sequence of operations to solve the problem. "
        "Your first step should always be to set the self.small = 0 self.big = 0 via using reset_problem tool. you can only use the "
        "reset_problem tool once as a first step."
    )

    depth = "Keep in mind the depth, the depth is the number of steps to reach the goal. For now try to get exactly 4 gallons in the 5-gallon jug with depth of {a} max."
    depth_value = depth.format(a=7)

    prompt += depth_value
    while True:
        response = agent.invoke(prompt)

        if problem.state()[1] == 4:
            for index, action in enumerate(response["intermediate_steps"]):
                index += 1
                print(f"{index}.{action[0].tool}")

            break


if __name__ == "__main__":
    main()