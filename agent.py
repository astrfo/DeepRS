class Agent:
    def __init__(self, policy):
        self.policy = policy
        self.current_state = None
        self.current_action = None

    def initialize(self):
        self.policy.reset()

    def action(self, state):
        action = self.policy.action(state)
        self.current_state = state
        self.current_action = action
        return action

    def greedy_action(self, state):
        action = self.policy.greedy_action(state)
        return action

    def update(self, state, action, reward, next_state, done):
        self.policy.update(state, action, reward, next_state, done)
