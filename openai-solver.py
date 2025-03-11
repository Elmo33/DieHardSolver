from langchain.tools import BaseTool
from langchain_openai import OpenAI
from typing import Optional, Type

class DieHardState:
    def __init__(self):
        self.small = 0  # 3-liter jug
        self.big = 0  # 5-liter jug

    def fill_small(self):
        print("Filling small jug")
        self.small = 3

    def fill_big(self):
        print("Filling big jug")
        self.big = 5

    def empty_small(self):
        print("Emptying small jug")
        self.small = 0

    def empty_big(self):
        print("Emptying big jug")
        self.big = 0

    def pour_small_into_big(self):
        print("Pouring small jug into big jug")
        old_big = self.big
        self.big = min(5, self.big + self.small)
        self.small -= (self.big - old_big)

    def pour_big_into_small(self):
        print("Pouring big jug into small jug")
        old_small = self.small
        self.small = min(3, self.small + self.big)
        self.big -= (self.small - old_small)

    def check_invariants(self):
        print("Checking invariants")
        assert 0 <= self.small <= 3, "Small jug out of bounds!"
        assert 0 <= self.big <= 5, "Big jug out of bounds!"
        return self.big != 4  # Problem solved if big jug has 4 liters


class DieHardTool(BaseTool):
    name: str = "die_hard_solver"  # Add explicit type annotation
    description: str = "A tool to manipulate water in two jugs to solve the Die Hard problem."
    state: DieHardState = DieHardState()  # Define the state properly

    def _run(self, action: str, query: Optional[str] = None) -> str:
        actions = {
            "fill_small": self.state.fill_small,
            "fill_big": self.state.fill_big,
            "empty_small": self.state.empty_small,
            "empty_big": self.state.empty_big,
            "pour_small_into_big": self.state.pour_small_into_big,
            "pour_big_into_small": self.state.pour_big_into_small,
        }

        if action in actions:
            actions[action]()
            self.state.check_invariants()
            return f"Action performed: {action}, New state: small={self.state.small}, big={self.state.big}"
        return "Invalid action!"

    def _arun(self, action: str, query: Optional[str] = None):
        raise NotImplementedError("Async execution is not supported.")

# Define LLM and tool
llm = OpenAI()
tool = DieHardTool()

# Example usage: Executing actions to solve the problem
sequence = [
    "fill_big", "pour_big_into_small", "empty_small",
    "pour_big_into_small", "fill_big", "pour_big_into_small"
]

for action in sequence:
    print(tool.run(action))
