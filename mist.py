from llama_cpp import Llama
import json
from main import WaterJugSolver
import re


llm = Llama.from_pretrained(
	repo_id="Qwen/Qwen1.5-7B-Chat-GGUF", filename="qwen1_5-7b-chat-q2_k.gguf",
)


# Tool definitions (Actions)
tools = {
    "fill_jug_A": lambda s, ca, cb: (ca, s[1]),
    "fill_jug_B": lambda s, ca, cb: (s[0], cb),
    "empty_jug_A": lambda s, ca, cb: (0, s[1]),
    "empty_jug_B": lambda s, ca, cb: (s[0], 0),
    "transfer_A_to_B": lambda s, ca, cb: (
        s[0] - min(s[0], cb - s[1]), s[1] + min(s[0], cb - s[1])
    ),
    "transfer_B_to_A": lambda s, ca, cb: (
        s[0] + min(s[1], ca - s[0]), s[1] - min(s[1], ca - s[0])
    ),
}




def generate_llm_action(state, ca, cb):
    prompt = f"""
Current state:
- Jug A: {state[0]} liters
- Jug B: {state[1]} liters
Capacities: A={ca}, B={cb}
Goal: Exactly 4 liters.

Reply with ONLY ONE word from the following:
fill_jug_A, fill_jug_B, empty_jug_A, empty_jug_B, transfer_A_to_B, transfer_B_to_A
Do NOT explain or add extra text.
"""

    while True:
        response = llm(prompt, max_tokens=10, temperature=0.0)
        text_response = response["choices"][0]["text"].strip()

        match = re.fullmatch(r"(fill_jug_A|fill_jug_B|empty_jug_A|empty_jug_B|transfer_A_to_B|transfer_B_to_A)", text_response)
        if match:
            return match.group(1)
        else:
            print(f"Unexpected response '{text_response}', retrying...")



def solve_with_llm(ca, cb, target):
    solver = WaterJugSolver(ca, cb, target)
    state = (0, 0)

    for step in range(400):
        action = generate_llm_action(state, ca, cb)
        print(f"Step {step + 1}: LLM chose '{action}', State: {state}")

        state = tools[action](state, ca, cb)

        if state[0] == target or state[1] == target:
            print(f"Target achieved at step {step + 1}! Final state: {state}")
            return

    print("Could not achieve target within step limit.")


# Run solver
solve_with = lambda ca, cb, target: solve_with_llm(ca, cb, target)

# Example call
solve_with_llm(3, 5, 4)
