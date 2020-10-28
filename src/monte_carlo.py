from src import environment
from statistics import mean
import random


class MonteCarloWithoutES:
    def __init__(self, epsilon: float, gamma: float):
        self.epsilon = epsilon
        self.gamma = gamma

    # policy evaluation
    def __evaluate_policy(self, policy: list) -> list:
        # start at given entry point
        state = environment.entry_id

        # choose random action
        action = random.choice(environment.labyrinth[state])

        sar_list = []  # (state, action, reward) tuple
        # play until win/loose
        while True:
            state, reward = environment.move(action, state)
            if reward is not 0:
                # game finished
                sar_list.append((state, None, reward))
                break
            else:
                # exploration factor epsilon
                action = random.choice(environment.get_valid_directions(state)) if random.random() <= self.epsilon else policy[state]
                sar_list.append((state, action, reward))

        # calculate total reward for each step of the episode
        g = 0
        sag_list = []  # (state, action, discounted reward) tuple
        sar_list.reverse()
        for s, a, r in sar_list:
            g = r + self.gamma * g
            sag_list.append((s, a, g))
        sag_list.reverse()
        return sag_list

    # policy improvement
    def generate_monte_carlo_policy(self, episodes: int) -> list:
        # create random policy
        policy = [random.choice(environment.labyrinth[state]) for state in environment.labyrinth]

        # init Q-Values and rewards list
        r = [{action: [0] for action in environment.labyrinth[field]} for field in environment.labyrinth]
        q_values = [{action: 0 for action in environment.labyrinth[field]} for field in environment.labyrinth]

        for i in range(episodes):
            sag_list = self.__evaluate_policy(policy)
            visited_states_actions = []  # (state, action) tuple

            # update Q-Values for each pair (s, a) of sag_list according to first visit
            for s, a, g in sag_list:
                if (s, a) not in visited_states_actions:
                    if a is not None:
                        r[s][a].append(g)
                        q_values[s][a] = mean(r[s][a])

            # update policy greedy
            for state in range(len(policy)):
                policy[state] = max(q_values[state], key=q_values[state].get)
        return policy


if __name__ == "__main__":
    mc_control = MonteCarloWithoutES(epsilon=0.4, gamma=0.9)
    mc_policy = mc_control.generate_monte_carlo_policy(10)
