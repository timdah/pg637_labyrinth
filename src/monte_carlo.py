import abc
import math
import sys

from src import environment
from statistics import mean
import random


class MonteCarlo:
    def __init__(self, gamma: float):
        self.gamma = gamma
        self.actions = list(environment.next_position_functions.keys())

    @abc.abstractmethod
    def _evaluate_policy(self, policy: list, episodes: int, current_episode: int) -> tuple:
        """
        Implement policy based on method used
        :param policy: Policy to evaluate
        :return: list of tuples (state, action, discounted reward)
        """

    # policy improvement
    def generate_monte_carlo_policy(self, episodes: int) -> tuple:
        # create random policy
        policy = [random.choice(self.actions) for _ in environment.position_ids]

        # init Q-Values and rewards list
        r = [{action: [] for action in self.actions} for _ in environment.position_ids]
        q_pi = [{action: 0 for action in self.actions} for _ in environment.position_ids]

        wins = loses = 0

        for i in range(episodes):
            sag_list, win = self._evaluate_policy(policy, episodes, i)
            if win:
                wins = wins + 1
            else:
                loses = loses + 1
            self.progress(i + 1, episodes, wins, loses)
            visited_states_actions = []  # (state, action) tuple

            # update Q-Values for each pair (s, a) of sag_list according to first visit
            for s, a, g in sag_list:
                if (s, a) not in visited_states_actions:
                    if a is not None:
                        r[s][a].append(g)
                        q_pi[s][a] = mean(r[s][a])

            # update policy greedy
            for state in range(len(policy)):
                policy[state] = max(q_pi[state], key=q_pi[state].get)
        v_pi = [max(q_pi[state].values()) for state in environment.position_ids]
        # for state in q_pi:
        #     print(len(state))
        return policy, v_pi

    def _calculate_sag_list(self, sar_list: list) -> list:
        g = 0
        sag_list = []  # (state, action, discounted reward) tuple
        sar_list.reverse()
        for s, a, r in sar_list:
            g = r + self.gamma * g
            sag_list.append((s, a, g))
        sag_list.reverse()
        return sag_list

    @staticmethod
    def progress(count, total, wins, loses):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        finish = ' - Done!\r\n' if count == total else ''

        sys.stdout.write(f'\r[{bar}] {count}/{total} Episodes - Wins: {wins} / Loses: {loses}{finish}')
        sys.stdout.flush()


class MonteCarloWithoutES(MonteCarlo):
    """
    On-policy first-visit MC control with epsilon-greedy policy
    """
    def __init__(self, epsilon: float, gamma: float):
        super().__init__(gamma)
        self.epsilon = epsilon
        self.annealing = True
        if self.annealing:
            self.epsilon_init = self.epsilon

    # policy evaluation
    def _evaluate_policy(self, policy: list, episodes: int, current_episode: int) -> tuple:
        # annealing epsilon
        if self.annealing:
            self.epsilon = self.epsilon_init * (1 - math.pow(current_episode/episodes, 2))

        # start at given entry point
        state = environment.entry_id

        # choose random action
        action = random.choice(self.actions)

        sar_list = []  # (state, action, reward) tuple
        win = None
        # play until win/loose
        while True:
            state, reward = environment.move(action, state)
            if reward is 1 or reward is -1:
                # game finished
                sar_list.append((state, None, reward))
                win = reward > 0
                break
            else:
                # exploration factor epsilon
                action = random.choice(self.actions) if random.random() <= self.epsilon else policy[state]
                sar_list.append((state, action, reward))
        # calculate total reward for each step of the episode
        return self._calculate_sag_list(sar_list), win


class MonteCarloExploringStart(MonteCarlo):
    """
    Not working because of deterministic policy.
    TODO make policy probabilistic
    """
    def __init__(self, gamma: float):
        super().__init__(gamma)

    @staticmethod
    def get_random_start_position():
        possible_states = list(environment.labyrinth.keys())
        possible_states.remove(environment.exit_id)
        possible_states.remove(environment.trap_id)
        return random.choice(possible_states)

    # policy evaluation
    def _evaluate_policy(self, policy: list, episodes: int, current_episode: int) -> list:
        # start at a random point (not exit or trap)
        state = self.get_random_start_position()

        # choose random action
        action = random.choice(environment.labyrinth[state])

        sar_list = []  # (state, action, reward) tuple
        # play until win/loose
        while True:
            state, reward = environment.move(action, state)
            print(f"({state}, {reward})")
            if reward is not 0:
                # game finished
                sar_list.append((state, None, reward))
                break
            else:
                # no exploration
                action = policy[state]
                sar_list.append((state, action, reward))
        print(f"sar_list: {sar_list}")

        # calculate total reward for each step of the episode
        return self._calculate_sag_list(sar_list)


if __name__ == "__main__":
    mc_control = MonteCarloWithoutES(epsilon=0.9, gamma=0.9)
    mc_policy, state_values = mc_control.generate_monte_carlo_policy(episodes=100)
    print(mc_policy)
    environment.prettyprint(state_values)
