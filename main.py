from langchain.tools import BaseTool
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from typing import NamedTuple

goal = 1

class JugState(NamedTuple):
    small_jug: int
    big_jug: int

    def as_dict(self) -> Dict[str, int]:
        return {"small_jug": self.small_jug, "big_jug": self.big_jug}

    @staticmethod
    def from_dict(info: Dict[str, int]) -> 'JugState':
        return JugState(info.get("small_jug", 0), info.get("big_jug", 0))


def enforce_constraints(state: JugState) -> JugState:
    if not (0 <= state.small_jug <= 3 and 0 <= state.big_jug <= 5):
        raise AssertionError("Invalid state: water level outside permitted limits.")
    if state.big_jug == goal:
        raise AssertionError(f"Error: The large jug contains {goal} liters!")
    return state

# Single class that handles all jug operations.
class WaterJugTool(BaseTool):
    name: str = "water_jug_tool"
    description: str = "A tool to manipulate water in two jugs to solve the Die Hard problem."

    def _run(self, input: Dict[str, Any]) -> Dict[str, int]:
        action = input.get("action")
        info = input.get("info", {"small_jug": 0, "big_jug": 0})
        current = JugState.from_dict(info)

        if action == "refill_small":
            print("DEBUG: Refilling small jug")
            updated_state = JugState(3, current.big_jug)
        elif action == "refill_big":
            print("DEBUG: Refilling big jug")
            updated_state = JugState(current.small_jug, 5)
        elif action == "drain_small":
            print("DEBUG: Draining small jug")
            updated_state = JugState(0, current.big_jug)
        elif action == "drain_big":
            print("DEBUG: Draining big jug")
            updated_state = JugState(current.small_jug, 0)
        elif action == "transfer_small_to_big":
            print("DEBUG: Transferring from small to big jug")
            room_in_big = 5 - current.big_jug
            amount = min(current.small_jug, room_in_big)
            updated_state = JugState(current.small_jug - amount, current.big_jug + amount)
        elif action == "transfer_big_to_small":
            print("DEBUG: Transferring from big to small jug")
            room_in_small = 3 - current.small_jug
            amount = min(current.big_jug, room_in_small)
            updated_state = JugState(current.small_jug + amount, current.big_jug - amount)
        elif action == "fetch_state":
            updated_state = current
        else:
            raise ValueError(f"Unknown action: {action}")

        return enforce_constraints(updated_state).as_dict()

    async def _arun(self, input: Dict[str, Any]) -> Dict[str, int]:
        return self._run(input)

# language_model = ChatOpenAI()
language_model = ChatOpenAI(model="gpt-4o")
tool = WaterJugTool()
model_with_tools = language_model.bind_tools([tool])

def instruction() -> str:
    msg = (
        "Your challenge is to solve a water jug puzzle by using the available water jug tool. "
        "Provide an input with an 'action' key (choose one from 'refill_small', 'refill_big', 'drain_small', 'drain_big', "
        "'transfer_small_to_big', 'transfer_big_to_small', 'fetch_state') and an 'info' key containing the current jug states. "
        f"Remember the jug constraints: small jug between 0-3 liters, big jug between 0-5 liters, and the big jug must never contain {goal} liters. "
        f"Continue applying operations until the error ('Error: The large jug contains {goal} liters!') is triggered."
    )
    return msg

def execute():
    conversation = [HumanMessage(instruction())]
    response = model_with_tools.invoke(conversation)
    current_state = JugState(0, 0)

    while response.tool_calls:
        conversation.append(response)
        for call in response.tool_calls:
            parameters = call["args"]
            if parameters.get("info") is None:
                parameters["info"] = {"small_jug": current_state.small_jug, "big_jug": current_state.big_jug}

            try:
                result = tool._run(parameters)
                current_state = JugState.from_dict(result)
            except Exception as err:
                result = str(err)

            print(f"Result: {result}")
            conversation.append(ToolMessage(result, tool_call_id=call["id"]))
        response = model_with_tools.invoke(conversation)
    conversation.append(response)
    print(response)

execute()

