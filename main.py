class WaterJugSolver:
    def __init__(self, capacity_A, capacity_B, target):
        self.capacity_A = capacity_A
        self.capacity_B = capacity_B
        self.target = target

    def fill_jug_A(self, state):
        return (self.capacity_A, state[1])

    def fill_jug_B(self, state):
        return (state[0], self.capacity_B)

    def empty_jug_A(self, state):
        return (0, state[1])

    def empty_jug_B(self, state):
        return (state[0], 0)

    def transfer_A_to_B(self, state):
        transfer_amount = min(state[0], self.capacity_B - state[1])
        return (state[0] - transfer_amount, state[1] + transfer_amount)

    def transfer_B_to_A(self, state):
        transfer_amount = min(state[1], self.capacity_A - state[0])
        return (state[0] + transfer_amount, state[1] - transfer_amount)

    def is_goal_reached(self, state):
        return state[0] == self.target or state[1] == self.target
