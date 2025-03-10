from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import OpenAI

import logging

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger("DieHardSolver")


class JugTool:
    def __init__(self, capacity):
        self.capacity = capacity
        self.current = 0

    def fill(self):
        logger.debug(f"Filling {self.capacity}-gallon jug to full capacity.")
        self.current = self.capacity
        return f"Filled {self.capacity}-gallon jug to {self.current} gallons."

    def empty(self):
        logger.debug(f"Emptying {self.capacity}-gallon jug.")
        self.current = 0
        return f"Emptied {self.capacity}-gallon jug."

    def transfer_to(self, other):
        amount = min(self.current, other.capacity - other.current)
        logger.debug(f"Transferring {amount} gallons from {self.capacity}-gallon jug "
                     f"to {other.capacity}-gallon jug.")
        self.current -= amount
        other.current += amount
        return (f"Transferred {amount} gallons. Now {self.capacity}-gallon jug has "
                f"{self.current} gallons, and {other.capacity}-gallon jug has "
                f"{other.current} gallons.")


# Define the JugTool objects
five_gallon = JugTool(5)
three_gallon = JugTool(3)

# Define the tools, ignoring arbitrary arguments
tools = [
    Tool(
        name="Fill 5-gallon Jug",
        func=lambda *args, **kwargs: five_gallon.fill(),
        description="Fill the 5-gallon jug completely."
    ),
    Tool(
        name="Fill 3-gallon Jug",
        func=lambda *args, **kwargs: three_gallon.fill(),
        description="Fill the 3-gallon jug completely."
    ),
    Tool(
        name="Empty 5-gallon Jug",
        func=lambda *args, **kwargs: five_gallon.empty(),
        description="Empty the 5-gallon jug."
    ),
    Tool(
        name="Empty 3-gallon Jug",
        func=lambda *args, **kwargs: three_gallon.empty(),
        description="Empty the 3-gallon jug."
    ),
    Tool(
        name="Transfer from 5 to 3",
        func=lambda *args, **kwargs: five_gallon.transfer_to(three_gallon),
        description="Pour water from the 5-gallon jug to the 3-gallon jug."
    ),
    Tool(
        name="Transfer from 3 to 5",
        func=lambda *args, **kwargs: three_gallon.transfer_to(five_gallon),
        description="Pour water from the 3-gallon jug to the 5-gallon jug."
    ),
]

llm = OpenAI(temperature=0)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)


def main():
    prompt = "We have a 5-gallon jug and a 3-gallon jug. Please measure exactly 4 gallons in the 5-gallon jug."
    response = agent.run(prompt)
    print("\nAgent’s final answer:\n", response)
    print(type(response))


if __name__ == "__main__":
    main()
