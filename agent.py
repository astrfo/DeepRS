class Agent:
    def __init__(self, policy):
        self.policy = policy

    def reset(self):
        self.policy.reset()

    def action(self, state):
        action = self.policy.action(state)
        return action

    def update(self, state, action, reward, next_state, done):
        self.policy.update(state, action, reward, next_state, done)
        