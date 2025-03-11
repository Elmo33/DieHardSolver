from langchain.llms import LlamaCpp
from langchain.tools import BaseTool
from typing import Optional, ClassVar

# Initialize the Qwen local AI model
model_path = "/root/DieHardSolver/7B/qwen1_5-7b-chat-q2_k.gguf"
llm_qwen = LlamaCpp(
    model_path=model_path,
    n_ctx=1024,
    n_threads=16,
    n_batch=256,
    use_mlock=True,
    use_mmap=True
)

class DieHardState:
    def __init__(self):
        self.small = 0
        self.big = 0
        self.history = {(0, 0)}

    def apply_state_change(self, new_small: int, new_big: int, action_name: str) -> bool:
        new_state = (new_small, new_big)
        if new_state in self.history:
            print(f"DEBUG: {action_name} would lead to repeated state: {new_state}. Avoid loops!")
            return False
        self.small = new_small
        self.big = new_big
        self.history.add(new_state)
        return True

    def fill_small(self):
        print(f"DEBUG: Filling small jug -> small={self.small}, big={self.big}")
        if self.small < 3:
            return self.apply_state_change(3, self.big, "fill_small")
        return False

    def fill_big(self):
        print(f"DEBUG: Filling big jug -> small={self.small}, big={self.big}")
        if self.big < 5:
            return self.apply_state_change(self.small, 5, "fill_big")
        return False

    def empty_small(self):
        print(f"DEBUG: Emptying small jug -> small={self.small}, big={self.big}")
        if self.small > 0:
            return self.apply_state_change(0, self.big, "empty_small")
        return False

    def empty_big(self):
        print(f"DEBUG: Emptying big jug -> small={self.small}, big={self.big}")
        if self.big > 0:
            return self.apply_state_change(self.small, 0, "empty_big")
        return False

    def pour_small_into_big(self):
        print(f"DEBUG: Pouring small into big -> small={self.small}, big={self.big}")
        if self.small > 0 and self.big < 5:
            amount = min(self.small, 5 - self.big)
            return self.apply_state_change(self.small - amount, self.big + amount, "pour_small_into_big")
        return False

    def pour_big_into_small(self):
        print(f"DEBUG: Pouring big into small -> small={self.small}, big={self.big}")
        if self.big > 0 and self.small < 3:
            amount = min(self.big, 3 - self.small)
            return self.apply_state_change(self.small + amount, self.big - amount, "pour_big_into_small")
        return False

    def check_goal(self):
        """Returns True if we reached the goal (big jug has 4 liters)."""
        return self.big == 4

    def get_valid_actions(self):
        """Return a list of actions that yield a new state (i.e. valid and not repeated)."""
        valid = []
        if self.small < 3 and (3, self.big) not in self.history:
            valid.append("fill_small")
        if self.big < 5 and (self.small, 5) not in self.history:
            valid.append("fill_big")
        if self.small > 0 and (0, self.big) not in self.history:
            valid.append("empty_small")
        if self.big > 0 and (self.small, 0) not in self.history:
            valid.append("empty_big")
        if self.small > 0 and self.big < 5:
            amount = min(self.small, 5 - self.big)
            new_state = (self.small - amount, self.big + amount)
            if new_state not in self.history and new_state != (self.small, self.big):
                valid.append("pour_small_into_big")
        if self.big > 0 and self.small < 3:
            amount = min(self.big, 3 - self.small)
            new_state = (self.small + amount, self.big - amount)
            if new_state not in self.history and new_state != (self.small, self.big):
                valid.append("pour_big_into_small")
        return valid

class DieHardTool(BaseTool):
    name: str = "die_hard_solver"
    description: str = "A tool to manipulate water in two jugs to solve the Die Hard problem using Qwen local AI."
    state: DieHardState = DieHardState()
    # Use the Qwen local AI model we defined above.
    llm: ClassVar = llm_qwen

    def _run(self, action: Optional[str] = None) -> str:
        if self.state.check_goal():
            return f"Goal reached! Final state: small={self.state.small}, big={self.state.big}"

        actions = {
            "fill_small": self.state.fill_small,
            "fill_big": self.state.fill_big,
            "empty_small": self.state.empty_small,
            "empty_big": self.state.empty_big,
            "pour_small_into_big": self.state.pour_small_into_big,
            "pour_big_into_small": self.state.pour_big_into_small,
        }

        if action:
            if action in actions:
                valid = actions[action]()
                if not valid:
                    return f"Action '{action}' resulted in no change or would lead to a repeated state. Try something else."
                return f"Action performed: {action}, New state: small={self.state.small}, big={self.state.big}"
            return "Invalid action!"

        valid_actions = self.state.get_valid_actions()

        prompt = f"""
You are solving the "Die Hard" water jug problem. Your goal is to measure exactly 4 liters in the big jug in the least amount of steps.

THE MOST IMPORTANT RULE: ONLY REPLY WITH THE ACTION THAT IS DEFINED IN THE RULES BELOW
**RULES:**
- NEVER repeat a previous state.
- Avoid actions that result in no change.
- Only select actions from the list below.

**Valid Actions Available Now:**
{valid_actions}

**Current State:**
- Small jug: {self.state.small} liters
- Big jug: {self.state.big} liters

**Previously Visited States:**
{list(self.state.history)}

Reply with only one of the valid actions above. Do not explain your choice.
"""
        next_action = self.llm.invoke(prompt).strip()

        if next_action not in valid_actions:
            return "Invalid action from LLM!"
        return self._run(next_action)

# Instantiate and run the Die Hard problem solver
tool = DieHardTool()

while not tool.state.check_goal():
    print(tool._run())
