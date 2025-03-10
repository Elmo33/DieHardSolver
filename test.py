from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import OpenAI
import logging

# Setup logging for debugging
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
        logger.debug(f"Transferring {amount} gallons from {self.capacity}-gallon jug to {other.capacity}-gallon jug.")
        self.current -= amount
        other.current += amount
        return f"Transferred {amount} gallons. Now {self.capacity}-gallon jug has {self.current} gallons, and {other.capacity}-gallon jug has {other.current} gallons."


# Define the tools
five_gallon = JugTool(5)
three_gallon = JugTool(3)

tools = [
    Tool(name="Fill 5-gallon Jug", func=five_gallon.fill, description="Fill the 5-gallon jug completely."),
    Tool(name="Fill 3-gallon Jug", func=three_gallon.fill, description="Fill the 3-gallon jug completely."),
    Tool(name="Empty 5-gallon Jug", func=five_gallon.empty, description="Empty the 5-gallon jug."),
    Tool(name="Empty 3-gallon Jug", func=three_gallon.empty, description="Empty the 3-gallon jug."),
    Tool(name="Transfer from 5 to 3", func=lambda: five_gallon.transfer_to(three_gallon),
         description="Pour water from the 5-gallon jug to the 3-gallon jug."),
    Tool(name="Transfer from 3 to 5", func=lambda: three_gallon.transfer_to(five_gallon),
         description="Pour water from the 3-gallon jug to the 5-gallon jug.")
]

# Use LangChain agent to control the sequence of steps
llm = OpenAI(temperature=0)  # Replace with an actual LLM if needed
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


def solve_die_hard():
    steps = [
        five_gallon.fill,
        five_gallon.transfer_to,  # Transfer to three_gallon
        three_gallon.empty,
        five_gallon.transfer_to,  # Transfer remaining to three_gallon
        five_gallon.fill,
        five_gallon.transfer_to  # Transfer to three_gallon to get exactly 4 gallons in 5-gallon jug
    ]

    logger.debug("Starting Die Hard problem solution...")
    for step in steps:
        print(step(three_gallon) if step == five_gallon.transfer_to else step())


solve_die_hard()
